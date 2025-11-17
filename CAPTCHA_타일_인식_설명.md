# CAPTCHA 스타일 이미지 타일 인식 시스템

## 프로젝트 개요

이 프로젝트는 CAPTCHA와 유사한 이미지 타일 인식 시스템을 구현합니다.
96x96 픽셀 이미지를 3x3 그리드(총 9개의 타일)로 분할하고, 각 타일에서 특정 객체를 찾아내는 딥러닝 알고리즘입니다.

## 주요 기능

1. **CNN 기반 이미지 분류**: TensorFlow/Keras를 사용한 Convolutional Neural Network
2. **타일 분할 시스템**: 하나의 이미지를 9개의 타일로 자동 분할
3. **객체 인식**: 10가지 객체 분류 (비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭)
4. **시각화**: 인식 결과를 시각적으로 표시 (타겟 객체가 포함된 타일에 초록색 테두리)

## 사용된 기술 스택

### 라이브러리
- **TensorFlow/Keras**: 딥러닝 모델 구축 및 학습
- **NumPy**: 배열 및 행렬 연산
- **Matplotlib**: 데이터 시각화
- **OpenCV (cv2)**: 이미지 전처리
- **scikit-learn**: 데이터 분할

### 모델 구조

```
입력: 32x32x3 (RGB 이미지)
    ↓
Conv2D (32 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
Flatten
    ↓
Dense (64 units) + ReLU
    ↓
Dense (10 units) + Softmax
    ↓
출력: 10개 클래스 확률
```

## 데이터셋

**CIFAR-10 데이터셋** 사용:
- 훈련 데이터: 50,000장
- 테스트 데이터: 10,000장
- 이미지 크기: 32x32x3 (RGB)
- 클래스 수: 10개

### 클래스 목록
0. 비행기 (airplane)
1. 자동차 (automobile)
2. 새 (bird)
3. 고양이 (cat)
4. 사슴 (deer)
5. 개 (dog)
6. 개구리 (frog)
7. 말 (horse)
8. 배 (ship)
9. 트럭 (truck)

## 핵심 함수 설명

### 1. `split_image_to_tiles(image, tile_size=32)`
- **기능**: 96x96 이미지를 3x3 그리드로 분할
- **입력**: 96x96x3 이미지
- **출력**: 9개의 32x32x3 타일 배열

### 2. `visualize_tiles(tiles, predictions=None, target_class=None)`
- **기능**: 9개의 타일을 시각화하고 예측 결과 표시
- **특징**: 타겟 클래스가 포함된 타일에 초록색 테두리 표시

### 3. `classify_tiles(tiles, model, target_class)`
- **기능**: 9개의 타일을 분류하고 타겟 객체 감지
- **입력**: 타일 배열, 모델, 타겟 클래스
- **출력**: 예측 결과, 타겟이 포함된 타일 인덱스

### 4. `create_captcha_image(x_data, y_data, target_class, num_targets)`
- **기능**: 테스트용 CAPTCHA 이미지 생성
- **특징**: 특정 객체를 원하는 개수만큼 포함시킴

### 5. `evaluate_captcha_system(model, x_data, y_data, num_tests)`
- **기능**: CAPTCHA 시스템의 정확도 평가
- **방법**: 여러 번의 테스트를 통해 정확도 계산

## 사용 방법

### 1. 모델 학습
```python
# 모델 생성
model = keras.models.Sequential([...])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

### 2. CAPTCHA 이미지 생성 및 테스트
```python
# CAPTCHA 이미지 생성 (자동차 2개 포함)
target_class = 1  # 자동차
captcha_img, true_labels = create_captcha_image(x_test, y_test,
                                                 target_class=target_class,
                                                 num_targets=2)

# 타일로 분할
tiles = split_image_to_tiles(captcha_img, tile_size=32)

# 분류 및 타겟 찾기
predictions, target_tiles = classify_tiles(tiles, model, target_class)

# 결과 시각화
visualize_tiles(tiles, predictions, target_class)
```

### 3. 커스텀 이미지로 테스트
```python
# 커스텀 이미지 로드
custom_img = load_custom_image('your_image.png')

# 타일 분할
custom_tiles = split_image_to_tiles(custom_img, tile_size=32)

# 분류
custom_predictions, custom_target_tiles = classify_tiles(custom_tiles, model, target_class=1)

# 시각화
visualize_tiles(custom_tiles, custom_predictions, target_class=1)
```

## 알고리즘 작동 원리

1. **전처리**
   - 이미지를 0~1 범위로 정규화
   - 96x96 이미지를 9개의 32x32 타일로 분할

2. **예측**
   - 각 타일을 CNN 모델에 입력
   - 각 타일에 대해 10개 클래스의 확률 계산

3. **타겟 감지**
   - 각 타일의 예측 클래스를 타겟 클래스와 비교
   - 타겟 클래스와 일치하는 타일의 인덱스 반환

4. **시각화**
   - 모든 타일을 3x3 그리드로 표시
   - 타겟이 포함된 타일에 초록색 테두리 표시
   - 각 타일의 예측 클래스와 확률 표시

## 모델 성능

- **테스트 정확도**: 약 70-75% (10 epochs 학습 시)
- **CAPTCHA 시스템 정확도**: 약 60-70% (완벽히 일치하는 경우)

## 개선 가능한 부분

1. **더 많은 학습**
   - epochs 수를 늘려 모델 성능 향상
   - Data Augmentation 적용

2. **더 복잡한 모델**
   - ResNet, VGG 등 사전 학습된 모델 사용
   - Transfer Learning 적용

3. **임계값 조정**
   - 확률 임계값을 설정하여 신뢰도 높은 예측만 사용

4. **앙상블 모델**
   - 여러 모델의 예측을 결합하여 정확도 향상

## 실전 활용

이 시스템은 다음과 같은 용도로 활용할 수 있습니다:

1. **웹사이트 보안**: 봇 방지용 CAPTCHA 시스템
2. **이미지 분류**: 대량의 이미지에서 특정 객체 찾기
3. **품질 검사**: 제품 이미지에서 결함 탐지
4. **교육용**: 딥러닝 및 이미지 처리 학습 자료

## 파일 구조

```
경승민_이미지_타일_인식_CAPTCHA(학생용).ipynb  # 메인 노트북 파일
CAPTCHA_타일_인식_설명.md                      # 이 설명서
captcha_tile_model.h5                         # 저장된 모델 (학습 후 생성)
```

## 참고 자료

- TensorFlow 공식 문서: https://www.tensorflow.org/
- CIFAR-10 데이터셋: https://www.cs.toronto.edu/~kriz/cifar.html
- Keras API: https://keras.io/

## 라이선스

이 프로젝트는 교육용 목적으로 작성되었습니다.

## 작성자

- 학번: 20901
- 이름: 경승민
- 작성일: 2024
