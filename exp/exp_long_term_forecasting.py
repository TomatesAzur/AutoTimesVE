from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        if self.args.use_multi_gpu:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        else:
            # 既存仕様を踏襲（self.device は run.py から渡る想定）
            self.device = self.args.gpu
            model = model.to(self.device)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print(n, p.dtype, p.shape)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        # “任意加工（セグメント平均/マスク重み等）を後段で可能にするため none に”
        criterion = nn.MSELoss(reduction='none')
        return criterion
    
    @staticmethod
    def reduce_loss_to_scalar(loss_map: torch.Tensor,
                              seg_size: int = None,
                              mask: torch.Tensor = None) -> torch.Tensor:
        """
        loss_map : [B,L,C] / [B,C] / [B,L] / [B] いずれにも対応
        seg_size: 変量セグメント幅（C を割り切るとき有効）例: 7
        mask    : 同形状 or ブロードキャスト可能な重み（任意）
        return  : スカラー Tensor
        """
        if not torch.is_tensor(loss_map):
            raise TypeError("loss_map must be a torch.Tensor")
        loss = loss_map.float()
        if mask is not None:
            loss = loss * mask.float()
        if loss.ndim == 0:
            return loss
        if loss.ndim == 1:  # [B]
            return loss.mean()
        if loss.ndim == 2:  # [B,L] or [B,C]
            return loss.mean()
        # [B,L,C]
        B, L, C = loss.shape
        if seg_size is not None and seg_size > 0 and (C % seg_size == 0):
            G = C // seg_size
            loss = loss.view(B, L, G, seg_size).mean(dim=3)  # セグメント内平均 → [B,L,G]
        return loss.mean()

    @torch.no_grad()
    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        """
        損失は「標準化スケール」かつ「未来のみ」で計算。
        標準化は各バッチの x(入力) の窓内統計（mean/std）で行う。
        """
        total_loss = []
        total_count = []
        time_now = time.time()
        test_steps = len(vali_loader)
        iter_count = 0
        self.model.eval()
        var_loss_sum = None  # [C]
        var_loss_batches = 0

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            iter_count += 1
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)  # 先に deviceへ
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs_raw = self.model(batch_x, batch_x_mark, None, batch_y_mark)  # 元スケール
            else:
                outputs_raw = self.model(batch_x, batch_x_mark, None, batch_y_mark)

            # 未来だけで比較する horizon を設定
            horizon = self.args.test_pred_len if is_test else self.args.token_len

            # x の窓内統計で outputs/batch_y を標準化して比較
            means = batch_x.mean(1, keepdim=True)
            stdev = (batch_x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
            outputs = (outputs_raw - means) / stdev
            batch_y_norm = (batch_y - means) / stdev

            outputs_ = outputs[:, -horizon:, :]
            batch_y_ = batch_y_norm[:, -horizon:, :]

            # 共通ヘルパでスカラーへ
            loss_map = criterion(outputs_, batch_y_)     # [B,H,C]
            loss = self.reduce_loss_to_scalar(loss_map, seg_size=getattr(self.args, 'seg_size', None))
            total_loss.append(loss.detach().cpu())
            total_count.append(batch_x.shape[0])

            # --- ★ 変量ごとの損失（z-score MSE） ---
            #     バッチ×時間平均 → [C]
            loss_by_var_batch = loss_map.mean(dim=(0, 1)).detach().cpu()  # [C]
            if var_loss_sum is None:
                var_loss_sum = loss_by_var_batch.clone()
            else:
                var_loss_sum += loss_by_var_batch
            var_loss_batches += 1

            if (i + 1) % 100 == 0:
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (test_steps - i)
                    print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                    iter_count = 0
                    time_now = time.time()

        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average([t.item() for t in total_loss], weights=total_count)).to(self.device)
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average([t.item() for t in total_loss], weights=total_count)

        # ★ 変量ごとの平均損失を表示（単GPU想定。DDP対応なら all_reduce 追加でOK）
        if var_loss_sum is not None and var_loss_batches > 0:
            mean_var_loss = (var_loss_sum / var_loss_batches).numpy()  # [C]
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print("---- Validation per-variable MSE (z-score scale, future horizon) ----")
                # vali_data は Dataset（VarTokensDataset）なので series_names を持つ
                var_names = getattr(vali_data, "series_names", None)
                if var_names is None:
                    # 名前がなければ index で表示
                    for c, v in enumerate(mean_var_loss):
                        print(f"  var[{c}]: {v:.6f}")
                else:
                    for name, v in zip(var_names, mean_var_loss):
                        print(f"  {name}: {v:.6f}")
        self.model.train()
        return total_loss

    def train(self, setting):
        _train_data, train_loader = self._get_data(flag='train')
        _vali_data,  vali_loader  = self._get_data(flag='val')
        _test_data,  test_loader  = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)
        
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        # 学習時の horizon は token_len に合わせる（元実装の流儀）
        train_horizon = self.args.token_len

        for epoch in range(self.args.train_epochs):
            iter_count = 0

            loss_val = torch.tensor(0., device=self.device)
            count = torch.tensor(0., device=self.device)
            
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                p_mask  = float(getattr(self.args, "mask_prob", 0.0))   # 0.0 なら再構成損オフ
                lam_rec = float(getattr(self.args, "lambda_rec", 0.0))

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs_raw = self.model(batch_x, batch_x_mark, None, batch_y_mark)  # 元スケール

                        # 標準化（xの窓内統計）＋ 未来だけ
                        means = batch_x.mean(1, keepdim=True)
                        stdev = (batch_x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
                        outputs = (outputs_raw - means) / stdev
                        batch_y_norm = (batch_y - means) / stdev

                        outputs_ = outputs[:, -train_horizon:, :]
                        batch_y_ = batch_y_norm[:, -train_horizon:, :]

                        # --- main loss ---
                        loss_map = criterion(outputs_, batch_y_)        # [B,H,C]
                        main_loss = self.reduce_loss_to_scalar(
                            loss_map, seg_size=getattr(self.args, 'seg_size', None)
                        )

                        total_loss = main_loss

                        # --- optional: masked cross-prediction reconstruction ---
                        if p_mask > 0.0 and lam_rec > 0.0:
                            B, H, C = outputs_.shape
                            # 変量単位マスク（時間方向に一貫）：[B,1,C] を H にブロードキャスト
                            var_mask = (torch.rand(B, 1, C, device=outputs_.device) < p_mask).float()
                            rec_elem = ((outputs_ - batch_y_) ** 2) * var_mask  # [B,H,C]
                            # 再利用: reduceヘルパ（セグメント平均した上で全平均）※分母の有効数正規化は不要でもOK
                            rec_loss = self.reduce_loss_to_scalar(
                                rec_elem, seg_size=getattr(self.args, 'seg_size', None)
                            )
                            total_loss = total_loss + lam_rec * rec_loss

                    scaler.scale(total_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                    loss = total_loss  # ログ用
                else:
                    outputs_raw = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    means = batch_x.mean(1, keepdim=True)
                    stdev = (batch_x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
                    outputs = (outputs_raw - means) / stdev
                    batch_y_norm = (batch_y - means) / stdev

                    outputs_ = outputs[:, -train_horizon:, :]
                    batch_y_ = batch_y_norm[:, -train_horizon:, :]

                    # --- main loss ---
                    loss_map = criterion(outputs_, batch_y_)  # [B,H,C]
                    main_loss = self.reduce_loss_to_scalar(
                        loss_map, seg_size=getattr(self.args, 'seg_size', None)
                    )
                    total_loss = main_loss

                    # --- optional: masked cross-prediction reconstruction ---
                    if p_mask > 0.0 and lam_rec > 0.0:
                        B, H, C = outputs_.shape
                        var_mask = (torch.rand(B, 1, C, device=outputs_.device) < p_mask).float()
                        rec_elem = ((outputs_ - batch_y_) ** 2) * var_mask  # [B,H,C]
                        rec_loss = self.reduce_loss_to_scalar(
                            rec_elem, seg_size=getattr(self.args, 'seg_size', None)
                        )
                        total_loss = total_loss + lam_rec * rec_loss

                    total_loss.backward()
                    model_optim.step()
                    loss = total_loss  # ログ用

                loss_val += loss.detach()
                count += 1
                
                if (i + 1) % 100 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))   
            if self.args.use_multi_gpu:
                dist.barrier()   
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / max(count.item(), 1.0)

            # ★ 検証/テスト損失も標準化スケール＆未来のみ（共通ヘルパでスカラー化）
            vali_loss = self.vali(_vali_data, vali_loader, criterion, is_test=False)
            test_loss = self.vali(_test_data, test_loader, criterion, is_test=True)

            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print("Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.use_multi_gpu:
                train_loader.sampler.set_epoch(epoch + 1)
                
        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        else:
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model

    def test(self, setting, test=0):
        # 既存と同じ：データ読み込み＆(必要なら)重みロード
        test_data, test_loader = self._get_data(flag='test')
        print("info:", self.args.test_seq_len, self.args.test_label_len, self.args.token_len, self.args.test_pred_len)
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            load_item = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)

        preds_std = []
        trues_std = []

        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 標準化の基準：最初の入力窓 batch_x の統計（学習/valiと同様）
                means0 = batch_x.mean(1, keepdim=True)  # [B,1,C]
                stdev0 = (batch_x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()

                # 自己回帰で test_pred_len 生成（元スケール）
                inference_steps = self.args.test_pred_len // self.args.token_len
                dis = self.args.test_pred_len - inference_steps * self.args.token_len
                if dis != 0:
                    inference_steps += 1
                pred_y_chunks = []
                for j in range(inference_steps):
                    if len(pred_y_chunks) != 0:
                        batch_x = torch.cat([batch_x[:, self.args.token_len:, :], pred_y_chunks[-1]], dim=1)
                        tmp = batch_y_mark[:, j-1:j, :]
                        batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)  # 元スケール
                    else:
                        outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    pred_y_chunks.append(outputs[:, -self.args.token_len:, :])

                pred_y = torch.cat(pred_y_chunks, dim=1)  # [B, test_pred_len, C]
                if dis != 0:
                    pred_y = pred_y[:, :-(self.args.token_len - dis), :]

                # 真値（未来部）
                true_y = batch_y[:, -self.args.test_pred_len:, :]

                # 標準化スケールで評価
                pred_std = (pred_y - means0) / stdev0
                true_std = (true_y - means0) / stdev0

                preds_std.append(pred_std.detach().cpu())
                trues_std.append(true_std.detach().cpu())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (test_steps - i)
                    print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # 既存の可視化（元スケール）
                if self.args.visualize and i == 0:
                    gt = true_y[0, :, -1].detach().cpu().numpy()
                    pd = pred_y[0, :, -1].detach().cpu().numpy()
                    lookback = batch_x[0, :, -1].detach().cpu().numpy()
                    gt = np.concatenate([lookback, gt], axis=0)
                    pd = np.concatenate([lookback, pd], axis=0)
                    dir_path = folder_path + f'{self.args.test_pred_len}/'
                    os.makedirs(dir_path, exist_ok=True)
                    visual(gt, pd, os.path.join(dir_path, f'{i}.png'))

        # ---- 指標の計算（標準化スケール）----
        preds_std = torch.cat(preds_std, dim=0).numpy()
        trues_std = torch.cat(trues_std, dim=0).numpy()
        mae, mse, rmse, mape, mspe = metric(preds_std, trues_std)
        print('[STD scale] mse:{}, mae:{}'.format(mse, mae))
        # ------------------------------
        # ★ 変量ごとの予測結果（標準化スケール）
        # ------------------------------
        C = preds_std.shape[-1]

        # Dataset 側に series_names があればそれを使う
        var_names = getattr(test_data, "series_names", None)
        if (var_names is None) or (len(var_names) != C):
            var_names = [f"var_{i}" for i in range(C)]

        per_var_results = []  # (name, mae, mse, rmse, mape, mspe) のリスト

        for idx, name in enumerate(var_names):
            # shape を [N, H, 1] にして metric をそのまま再利用
            mae_c, mse_c, rmse_c, mape_c, mspe_c = metric(
                preds_std[:, :, idx:idx+1],
                trues_std[:, :, idx:idx+1]
            )
            per_var_results.append((name, mae_c, mse_c, rmse_c, mape_c, mspe_c))
            print(f"[STD scale][{name}] mse:{mse_c}, mae:{mae_c}")

        # ★ CSV として保存（test_results/setting/...）
        csv_path = os.path.join(folder_path, f"{self.args.test_pred_len}_per_var_metrics.csv")
        with open(csv_path, "w") as f_csv:
            f_csv.write("var,mae,mse,rmse,mape,mspe\n")
            for name, mae_c, mse_c, rmse_c, mape_c, mspe_c in per_var_results:
                f_csv.write(f"{name},{mae_c},{mse_c},{rmse_c},{mape_c},{mspe_c}\n")

        # ログは標準化スケールを保存
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('[STD scale] mse:{}, mae:{}'.format(mse, mae))
        f.write('\n\n')
        f.close()
        return
