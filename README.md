## Torch7과 다수의 GPU를 사용한 [이미지네트(ImageNet)](http://image-net.org/download-images) 객체 분류

원문: https://github.com/soumith/imagenet-multiGPU.torch

(영상들을 위한 범용 그리고 고도로 확장 가능한 데이터 로더를 포함하는 1,200줄의) 이 짧은 예제로 우리는 다음을 보이려고 합니다:
- 이미지네트에 대해 [알렉스네트(AlexNet)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) 또는 [오버피트(Overfeat)](http://arxiv.org/abs/1312.6229)를 훈련
- 다양한 백엔드(backend)를 보여줌: CuDNN, CuNN
- 여러 개의 GPU로 훈련을 가속하기 위한 nn.ModelParallel과 nn.DataParalell 사용
- 훈련을 더 가속하기 위한 nn.SpatialConvolutionCuFFT 사용
- 디스크에서 멀티스레드로 데이터 로딩 (한 스레드에서 다른 한 스레드로 직렬화 없이 텐서들의 전송을 보임)

### 미리 준비할 것
- 쿠다(CUDA) GPU가 있는 피시(PC)에 토치(Torch) 배포판 설치
- http://image-net.org/download-images 에서 2012년 ImageNet 데이터세트 내려받기. 그 데이터세트는 1,000개 부류(class)와 120만 개 영상들로 구성됨.
- 이 명령어 실행:
```bash
git clone https://github.com/torch/nn && cd nn && git checkout getParamsByDevice && luarocks make rocks/nn-scm-1.rockspec
```

### 데이터 처리
이미지네트 훈련 영상들은 이미 n07579787, n07880968 같은 적절한 하위 폴더에 있습니다. 우리가 해야 할 일은 검증 정답(validation groundtruth)을 얻고 그 검증 영상들을 적절한 하위 폴더들로 옮기는 일입니다. 이를 위해, ILSVRC2012_img_train.tar 그리고 ILSVRC2012_img_val.tar를 내려받으십시오. 그리고 다음 명령어들을 입력합니다:

```bash
# 훈련 데이터 추출
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
# 검증 데이터 추출
cd ../ && mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

이제 모두 준비되었습니다!

만약 당신의 이미지네트 데이터세트가 하드디스크 또는 느린 SSD에 있다면, 모든 영상의 크기를 바꾸기 위해 이 명령어를 실행하십시오. 영상들이 256차원으로 더 작게 바뀌고 가로세로 비율은 그대로 유지될 것입니다. 이 명령어는 데이터가 디스크에서 더 빨리 로드되도록 돕습니다.

```bash
find . -name "*.JPEG" | xargs -I {} convert {} -resize "256^>" {}
```

### 실행
훈련 스크립트들에는 몇 가지 옵션들이 딸려 있습니다. 그 스크립트들을 --help 플래그와 함께 실행시키면 그 옵션들을 볼 수 있습니다. 
```bash
th main.lua --help
```

훈련을 실행하기 위해 main.lua를 실행합니다.
기본으로, main.lua 스크립트는 CuDNN과 두 개의 데이터-로더 스레드 기반 1-GPU 알렉스네트를 실행합니다.
```bash
th main.lua -data [train과 val 폴더가 들어있는 이미지네트 폴더]
```

2-GPU 기반 알렉스네트 + CuDNN을 실행하기 위해서는 이렇게 입력합니다:
```bash
th main.lua -data [train과 val 폴더가 들어있는 이미지네트 폴더] -nGPU 2 -backend cudnn -netType alexnet
```
비슷하게, 다른 쿠다 커널을 사용하기 위해 백엔드를 'cunn'으로 바꿀 수도 있습니다. 

또한 다음 명령어를 사용하여 오버피트를 훈련할 수 있습니다:
```bash
th main.lua -data [train과 val 폴더가 들어있는 이미지네트 폴더] -netType overfeat

# 다수의 GPU 오버피트 (2-GPU라고 합시다)
th main.lua -data [train과 val 폴더가 들어있는 이미지네트 폴더] -netType overfeat -nGPU 2
```

훈련 스크립트는 현재 top-1 그리고 top-5 에러와 매 미니 배치(mini-batch)의 목적 함수 손실(objective loss)를 출력합니다.
우리는 알렉스네트가 53 에포크(epoch)가 끝날 때 42.5% 에러로 수렴하도록 학습률을 고정하였습니다(hard-coded).

매 에포크 끝에서, 모델은 디스크에 model_[xx].t7과 같은 이름으로 저장됩니다. 그 이름에서 xx는 에포크 횟수입니다.
torch.load를 사용하여 이 모델은 언제든지 다시 토치로 로드될 수 있습니다.
```lua
model = torch.load('model_10.t7') -- 저장된 모델을 다시 로딩
```

유사하게, 만약 모델을 새로운 영상에 대해 시험하고 싶다면 그 영상을 로드하기 위해 donkey.lua의 103번째 줄에 있는 testHook를 사용할 수 있습니다. testHook는 예측을 위해 그 영상을 모델에 입력합니다. 이를테면:
```lua
dofile('donkey.lua')
img = testHook({loadSize}, 'test.jpg')
model = torch.load('model_10.t7')
predictions = model:forward(img:cuda())
```

만약 이 예제를 재사용하거나 스크립트를 디버그하기 원한다면 단일 스레드 모드로 디버그 및 개발하기를 추천합니다. 그래야 stack trace들이 완전히 출력되기 때문입니다.
```lua
th main.lua -nDonkeys 0 [...options...]
```

### 코드 설명
- `main.lua` (~30 줄) - 모든 다른 파일을 로드, 훈련 시작.
- `opts.lua` (~50 줄) - 모든 커맨드-라인 옵션과 설명
- `data.lua` (~60 줄) - 병렬 데이터 로딩을 위한 K 스레드 생성을 위한 로직(logic)을 포함.
- `donkey.lua` (~200 줄) - 데이터 로딩 로직과 그 세부사항을 포함. 이 파일은 각 데이터 로더 스레드에 의해 실행됨. 10 개의 잘린 영상들을 만드는 랜덤 영상 자르기(cropping) 등이 여기에 있음.
- `model.lua` (~80 줄) - 알렉스네트 모델과 판별 함수 생성.
- `train.lua` (~190 줄) - 네트워크를 훈련하기 위한 로직. 좋은 결과를 만들기 위해 학습률과 가중치 감소(wight decay)는 코드 내에서 고정됨(hard-coded).
- `test.lua` (~120 줄) - (top-1과 top-5 에러 계산을 포함하는) 검증 집합에서 네트워크를 시험하기 위한 로직.
- `dataset.lua` (~430 줄) - 범용 데이터 로더, 대부분 [여기: imagenetloader.torch](https://github.com/soumith/imagenetloader.torch)에서 파생됨. 이 로더 사용을 위한 더 많은 문서들과 예제들이 그 저장소에 있음.
