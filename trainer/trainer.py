import time
import torch
import torchvision.utils as vutils
from tqdm import tqdm

from base import BaseTrainer
from utils import WarmupPolyLR, runningScore, cal_text_score


class Trainer(BaseTrainer):
    def __init__(self, config, model, criterion, train_loader, validate_loader=None, post_process=None, metric_cls=None):
        super().__init__(config=config, model=model, criterion=criterion)

        self.train_loader = train_loader
        self.validate_loader = validate_loader
        self.post_process = post_process
        self.metric_cls = metric_cls

        if self.validate_loader is not None:
            assert self.post_process is not None and self.metric_cls is not None

        self.train_loader_len = len(self.train_loader)
        self.show_images_iter = config['trainer']['show_images_iter']

        self.scheduler = WarmupPolyLR(
            self.optimizer,
            max_iters=self.epochs * self.train_loader_len,
            warmup_iters=self.config['lr_scheduler']['args']['warmup_epoch'] * self.train_loader_len,
            **self.config['lr_scheduler']['args']
        )

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_start = time.time()
        running_metric_text = runningScore(2)
        total_loss = 0.0

        for i, batch in enumerate(tqdm(self.train_loader)):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)

            cur_batch_size = batch['img'].size(0)

            preds = self.model(batch['img'])
            loss_dict = self.criterion(preds, batch)
            loss = loss_dict['loss']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            score_shrink_map = cal_text_score(
                preds[:, 0, :, :],
                batch['shrink_map'],
                batch['shrink_mask'],
                running_metric_text,
                thred=self.config['post_processing']['args']['thresh']
            )

            train_loss = loss.item()
            total_loss += train_loss

            acc = score_shrink_map['Mean Acc']
            iou_shrink_map = score_shrink_map['Mean IoU']

            lr = max(self.scheduler.get_last_lr()[0], 1e-7)

            if self.global_step % self.log_iter == 0:
                self.logger_info(
                    f"[{epoch}/{self.epochs}] step:{self.global_step}/{self.train_loader_len}, "
                    f"loss: {train_loss:.4f}, acc: {acc:.4f}, iou: {iou_shrink_map:.4f}, lr: {lr:.8f}"
                )

            if self.tensorboard_enable and self.global_step % self.show_images_iter == 0:
                self.writer.add_scalar('Loss/train_loss', train_loss, self.global_step)
                self.writer.add_scalar('LR', lr, self.global_step)

            self.global_step += 1

        avg_loss = total_loss / len(self.train_loader)

        return {
            'epoch': epoch,
            'train_loss': avg_loss,
            'time': time.time() - epoch_start,
            'lr': lr
        }

    def _eval(self, epoch):
        self.model.eval()
        total_frame = 0.0
        total_time = 0.0
        raw_metrics = []

        with torch.no_grad():
            for batch in tqdm(self.validate_loader, desc=f'Evaluating Epoch {epoch}'):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)

                start_time = time.time()
                preds = self.model(batch['img'])
                boxes, scores = self.post_process(preds, batch['shape'], is_output_polygon=self.metric_cls.is_output_polygon)
                raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores))
                raw_metrics.append(raw_metric)

                total_frame += batch['img'].size(0)
                total_time += time.time() - start_time

        metrics = self.metric_cls.gather_measure(raw_metrics)
        recall = metrics['recall'].avg
        precision = metrics['precision'].avg
        hmean = metrics['fmeasure'].avg

        self.logger_info(f'Epoch {epoch} - Precision: {precision:.4f}, Recall: {recall:.4f}, Hmean: {hmean:.4f}, FPS: {total_frame / total_time:.2f}')

        return recall, precision, hmean

    def _on_epoch_finish(self):
        epoch = self.epoch_result['epoch']
        self._save_checkpoint(epoch, f'model_epoch_{epoch}.pth')

    def _on_train_finish(self):
        self.logger_info('Training Complete!')
        self.logger_info(f"Best Model: Epoch {self.metrics['best_model_epoch']} - hmean: {self.metrics['hmean']:.4f}")
