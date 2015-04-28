## [이미지네트(ImageNet)](http://image-net.org/download-images) 데이터에 대해 토치7(Torch7)과 여러 개의 GPU로 객체 분류 훈련하기

원문: https://github.com/soumith/imagenet-multiGPU.torch

(영상들을 위한 범용 그리고 고도로 확장 가능한 데이터 로더를 포함하는 1,200줄의) 이 짧은 예제를 통해 우리는 다음을 보이고자 합니다:
- 이미지네트에 대해 [알렉스네트(AlexNet)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) 또는 [오버피트(Overfeat)](http://arxiv.org/abs/1312.6229)를 훈련
- 다양한 백엔드(backend)들을 보여줌: CuDNN, CuNN
- 여러 개의 GPU들로 훈련을 가속하기 위해 nn.ModelParallel 그리고 nn.DataParalell 사용
- 훈련을 더욱 가속하기 위해 nn.SpatialConvolutionCuFFT 사용
- 디스크로부터의 멀티스레드화된 데이터 로딩 (한 스레드에서 한 다른 스레드로 직렬화 없이 텐서들의 전송을 보여줌)

### 미리 준비할 것들
- 쿠다(CUDA) GPU가 있는 피시(PC)에 토치(Torch) 배포판 설치
- http://image-net.org/download-images 에서 2012년 ImageNet 데이터세트 내려받기. 그 데이터세트는 1,000개 부류와 120만 개 영상들로 구성됨.
- 이 명령어 실행:
```bash
git clone https://github.com/torch/nn && cd nn && git checkout getParamsByDevice && luarocks make rocks/nn-scm-1.rockspec
```

### 데이터 처리
**어떤 데이터베이스에서도 영상들은 전처리되거나 패키징될(packaged) 필요가 없습니다.** 데이터베이스는 (보통 빠른 로드를 위해) [SSD](http://ko.wikipedia.org/wiki/%EC%86%94%EB%A6%AC%EB%93%9C_%EC%8A%A4%ED%85%8C%EC%9D%B4%ED%8A%B8_%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C)에 저장됩니다. 그러나 우리는 속도의 손실 없이 [NFS](http://ko.wikipedia.org/wiki/%EB%84%A4%ED%8A%B8%EC%9B%8C%ED%81%AC_%ED%8C%8C%EC%9D%BC_%EC%8B%9C%EC%8A%A4%ED%85%9C) 기반의 데이터 로더를 사용하였습니다. 사용법은 단순합니다: SubFolderName == ClassName. 예를 들어, 만약 {cat,dog} 부류가 있다면, cat 영상들은 dataset/cat 폴더로 그리고 dog 영상들은 dataset/dog 폴더로 갑니다.

이미지네트의 훈련 영상들은 이미 (n07579787, n07880968 같은) 적절한 하위 폴더들에 있습니다. 당신이 해야할 일은 검증 정답(validation groundtruth)을 얻고 그 검증 영상들을 적절한 하위 폴더들로 옮기는 것입니다. 이것을 하기 위해, ILSVRC2012_img_train.tar 그리고 ILSVRC2012_img_val.tar를 다운로드 받으십시오. 그리고 다음 명령어들을 사용하십시오:

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

만약 당신의 이미지네트 데이터세트가 하드디스크 또는 느린 SSD에 있다면, 모든 영상의 크기를 바꾸기 위해 이 명령어를 실행하십시오. 그 영상들은 더 작은 256차원으로 바뀌고 가로세로 비율은 그대로 유지됩니다. 이 명령어가 데이터가 디스크에서 더 빨리 로드되도록 도울 것입니다.

```bash
find . -name "*.JPEG" | xargs -I {} convert {} -resize "256^>" {}
```

### 실행
훈련 스크립트들에는 몇 가지 옵션들이 딸려 있습니다. 그 옵션들은 그 스크립트를 --help 플래그와 함께 실행시킴으로써 볼 수 있습니다. 
```bash
th main.lua --help
```

훈련을 실행하기 위해, main.lua를 실행시킵니다.
기본적으로, 그 스크립트는 CuDNN과 두 개의 데이터-로더 스레드 기반의 1-GPU 알렉스네트를 실행시킵니다.
```bash
th main.lua -data [train과 val 폴더가 들어있는 이미지네트 폴더]
```

2-GPU 기반 알렉스네트 + CuDNN을 위해서는, 이렇게 실행시킵니다:
```bash
th main.lua -data [train과 val 폴더가 들어있는 이미지네트 폴더] -nGPU 2 -backend cudnn -netType alexnet
```
유사하게, 다른 쿠다 커널들을 사용하기 위해 백엔드를 'cunn'으로 바꿀 수 있습니다. 

또한 당신은 다음 명령어를 사용하여 오버피트를 훈련시킬 수 있습니다:
```bash
th main.lua -data [train과 val 폴더가 들어있는 이미지네트 폴더] -netType overfeat

# 여러 개 GPU 오버피트 (2-GPU라고 합시다)
th main.lua -data [train과 val 폴더가 들어있는 이미지네트 폴더] -netType overfeat -nGPU 2
```

훈련 스크립트는 현재 top-1 그리고 top-5 에러와 매 미니 배치(mini-batch)에서의 목적 함수 손실(objective loss)를 출력합니다.
우리는 알렉스네트가 53 에포크(epoch)의 끝에서 42.5% 에러로 수렴하도록 학습률을 고정시켰습니다(hard-coded).

매 에포크의 끝에서, 모델은 디스크에 model_[xx].t7와 같은 이름으로 저장됩니다. 그 이름에서 xx는 에포크 횟수입니다.
당신은 torch.load를 사용하여 이 모델을 언제든지 다시 토치로 로드할 수 있습니다.
```lua
model = torch.load('model_10.t7') -- 저장된 모델을 다시 로딩
```

유사하게, 만약 당신의 모델을 새로운 영상에서 시험하고 싶다면, 당신의 영상을 로드하기 위해 donkey.lua의 103번째 줄에 있는 testHook를 사용할 수 있습니다. testHook은 예측을 위해 그 영상을 모델에 입력합니다. 이를테면:
```lua
dofile('donkey.lua')
img = testHook({loadSize}, 'test.jpg')
model = torch.load('model_10.t7')
predictions = model:forward(img:cuda())
```

만약 이 예제를 재사용하고, 스크립트들을 디버그하기 원한다면, 저는 싱글-스레드 모드로 디버그 및 개발하기를 추천합니다. 그래야 stack trace들이 완전히 출력되기 때문입니다.
```lua
th main.lua -nDonkeys 0 [...options...]
```

### 코드 설명
- `main.lua` (~30 줄) - 모든 다른 파일들을 로드, 훈련 시작.
- `opts.lua` (~50 줄) - 모든 커맨드-라인 옵션들과 설명
- `data.lua` (~60 줄) - 병렬 데이터 로딩을 위한 K 스레드들 생성을 위한 로직(logic)을 포함.
- `donkey.lua` (~200 줄) - 데이터 로딩 로직과 그 세부사항들을 포함. 이 파일을 각 데이터 로더 스레드에 의해 실행됨. 10 개의 잘린 영상들을 만드는 랜덤 영상 자르기(cropping) 등이 여기에 있음.
- `model.lua` (~80 줄) - 알렉스네트 모델과 판별 함수 생성
- `train.lua` (~190 줄) - 네트워크를 훈련시키기 위한 로직. 좋은 결과들을 만들기 위해 학습률과 가중치 디케이(wight decay)는 프로그램 내부적으로 고정됨(hard-coded).
- `test.lua` (~120 줄) - 검증 집합에서 네트워크를 시험하기 위한 로직 (top-1과 top-5 에러 계산을 포함하는)
- `dataset.lua` (~430 줄) - 범용 데이터 로더, 대부분 [여기: imagenetloader.torch](https://github.com/soumith/imagenetloader.torch)에서 파생됨. 그 저장소에 이 로더 사용을 위한 더 많은 문서들과 예제들이 있음.
