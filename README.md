# Attention과 Grad-CAM을 이용한 블랙박스 심층 신경망 모델의 시각적 설명 기법

### Contribution
* 블랙박스 모델의 내부를 reproducing 하므로 attention map을 다양하게 활용 가능
* Attention map이 정교함
* Adversarial example이 생기는 문제가 발생하지 않음
* Attention map 생성 속도가 빠름
   

### 제안하는 기법의 개요 및 특징
* 모델의 내부에 접근할 수 있으면 모델 예측을 해석하기 쉬움
* 블랙박스 모델과 동일한 예측을 할 수 있도록 reproducing된 모델 생성
  - Knowledge Distillation 기법 적용
* Attention map을 생성하는 모듈을 삽입해 향상된 돌출맵 생성
  - 전체 레이어에서 attention map 생성
* 특정 클래스를 예측할 때 중요한 영역 정보 포함하는 attention map 생성
  - 전체 레이어에 걸쳐 Grad-CAM 이용
  
### 제안하는 기법 구조
![스크린샷, 2020-06-01 18-08-55](https://user-images.githubusercontent.com/25657945/83394164-0efd7380-a433-11ea-82db-1f4fce73c48d.png)
![스크린샷, 2020-06-01 18-09-30](https://user-images.githubusercontent.com/25657945/83394167-0f960a00-a433-11ea-9848-7f72209d63f9.png)

###
