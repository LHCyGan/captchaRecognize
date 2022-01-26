# -*- encoding:utf-8 -*-
# author: liuheng
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from Dataset import ImageDataset
from utils import tensor_to_str, ctc_to_str
from torchvision import transforms as T
from module import LstmCTCNet



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 8
CAPTCHA_MAX_LENGTH = 5                       # 验证码长度
IMAGE_SHAPE = (64, 160)
# 4 * 10
# _a_b_c_d_e_  11  时序长度 最好是多少呢？ CAPTCHA_MAX_LENGTH * 2 + 1

train_set = ImageDataset(
    './data/train',
    maxLength=CAPTCHA_MAX_LENGTH,
    transform=T.Compose([
        T.ToPILImage(),
        T.Resize(IMAGE_SHAPE),
        T.ToTensor(),
        T.Normalize((0.79490087, 0.79427771, 0.79475806), (0.30808181, 0.30900241, 0.30821851))
    ])
)
valid_set = ImageDataset(
    './data/valid',
    maxLength=CAPTCHA_MAX_LENGTH,
    transform=T.Compose([
        T.ToPILImage(),
        T.Resize(IMAGE_SHAPE),
        T.ToTensor(),
        T.Normalize((0.79490087, 0.79427771, 0.79475806), (0.30808181, 0.30900241, 0.30821851))
    ])
)

CHAR_SET = train_set.get_label_map()
CAPTCHA_CHARS = len(CHAR_SET)                # 分类数

# print(CHAR_SET)
# print(train_set.get_mean_std())

train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    model = LstmCTCNet(IMAGE_SHAPE, CAPTCHA_CHARS)
    model = model.to(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.CTCLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3)

    for epoch in range(0, 100):
        # Train
        bar = tqdm(train_loader, 'Training')
        for images, texts, target_lengths in bar:
            images = images.to(DEVICE)

            optimizer.zero_grad()
            predict = model(images)

            predict_lengths = torch.IntTensor([int(predict.shape[0])] * texts.shape[0])


            loss = loss_func(predict, texts.to(DEVICE), predict_lengths, target_lengths.to(DEVICE))
            loss.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]['lr']
            bar.set_description(
                "Train epocch %d, loss %.4f, lr %.6f" % (epoch, loss.detach().cpu.numpy(), lr)
            )


        # Valid
        bar = tqdm(valid_loader, 'Validating')
        correct = count = 0
        for images, texts, target_lengths in bar:
            images = images.to(DEVICE)

            predicts = model(images)  # torch.Size([10, 128, 37])
            for i in range(predicts.shape[1]):
                predict = predicts[:, i, :]  # [10, 37]
                predict = predict.argmax(1)  # [10]
                predict = predict.contiguous()
                count += 1
                label_text = tensor_to_str(texts[i], CHAR_SET)[:target_lengths[i]]
                predict_text = ctc_to_str(predict, CHAR_SET)
                if label_text == predict_text:
                    correct += 1

            predict_lengths = torch.IntTensor([[int(predicts.shape[0])] * texts.shape[0]])

            loss = loss_func(predicts, texts.to(DEVICE), predict_lengths, target_lengths.to(DEVICE))

            lr = optimizer.param_groups[0]['lr']
            bar.set_description("Valid epoch %d, acc %.4f, loss %.4f, lr %.6f" % (
                epoch, correct / count, loss.detach().cpu().numpy(), lr
            ))

        scheduler.step(correct / count)
        torch.save(model.state_dict(), "models/save_%d.model" % epoch)

