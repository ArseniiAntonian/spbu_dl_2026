# One Piece classification 2026

18 персонажей, метрика — F1 (считаем macro), 2915 картинок в train, 849 в test.
Данные ожидаются в `../../one-piece-classification-2026/` (в репозиторий не входят).

## Пайплайн

| Шаг | Скрипт | Результат |
|---|---|---|
| Кэш: ресайз до 384 px, RGBA → RGB на белом | `prepare.py` | `outputs/cache_384.pkl` |
| Сравнение бэкбонов: замороженные признаки + логрег | `probe.py <timm-модели>` | `outputs/feats_*.npz` |
| 5 фолдов, почти-дубликаты (cos > 0.95 по CLIP) в одном фолде | `folds.py` | `outputs/folds.npy` |
| Fine-tune учителя (LP-FT: голова из линейной пробы) | `train_teacher.py --init-head` | `outputs/<run>/` |
| Логиты учителя на 10 детерминированных аугментациях train+test | `make_views.py` | `outputs/<name>_views0-10.npy` |
| Дистилляция в свою CNN (`student.py`, 1.18M параметров, 4.5 МБ) | `distill.py` | `outputs/<run>/student_ema.pt` |

Учитель на MPS слишком медленный (~130 img/s инференс), чтобы считать его на каждом шаге обучения ученика,
поэтому его выходы считаются заранее на фиксированных аугментациях, которые воспроизводятся по сиду.
Depthwise-свёртки на MPS в ~8 раз медленнее обычных, поэтому ученик — ResNet-подобная сеть с SE.

## Результаты (macro-F1 на фолде 0, дубликаты не протекают между фолдами)

| Модель | F1 |
|---|---|
| Линейная проба: CLIP ViT-B/16 | 0.970 |
| Линейная проба: DINOv2 ViT-B/14 | 0.902 |
| Линейная проба: SigLIP2 ViT-B/16 | 0.982 |
| Линейная проба: concat трёх | 0.980 |
| SigLIP2 обычный fine-tune, 5 эпох, lr 1e-4 | 0.958 |
| SigLIP2 LP-FT, 6 эпох, lr 3e-5, layer decay 0.75 | 0.990 |
| **LP-FT + линейная проба (0.5 / 0.5)** | **0.993** |

Сабмиты в `outputs/`: `sub_ens_lpft_linear.csv` — лучший на данный момент.

## Команды

```bash
python prepare.py
python probe.py vit_base_patch16_clip_224.openai vit_base_patch16_siglip_224.v2_webli vit_base_patch14_dinov2.lvd142m
python folds.py
python train_teacher.py --model vit_base_patch16_siglip_224.v2_webli --fold 0 --epochs 6 --lr 3e-5 --layer-decay 0.75 --init-head --run siglip_lpft_f0
python make_views.py --teachers siglip_lpft_f0 --views 10 --out lpft_f0
python distill.py --views lpft_f0_views0-10.npy --fold 0 --epochs 150 --img-size 160 --run student_f0
```

## Дальше

1. Ученик: прогон с `--fold 0` для честной оценки, затем `--fold -1` (весь train + test) для сабмита.
2. LP-FT SigLIP на всём train (`--fold -1`) и ансамбль с линейной пробой для финального сабмита.
