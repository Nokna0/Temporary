# -*- coding: utf-8 -*-
"""
CAPTCHA 타일 인식 시스템 - 간단한 데모 버전
학번: 20901, 이름: 경승민
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정 (matplotlib에서 한글 표시)
plt.rcParams['font.family'] = 'DejaVu Sans'


class CaptchaTileRecognizer:
    """CAPTCHA 타일 인식 시스템 클래스"""

    def __init__(self):
        """초기화"""
        self.model = None
        self.class_names = ['비행기', '자동차', '새', '고양이', '사슴',
                           '개', '개구리', '말', '배', '트럭']

    def build_model(self):
        """CNN 모델 생성"""
        model = keras.models.Sequential([
            # 첫 번째 합성곱 레이어
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),

            # 두 번째 합성곱 레이어
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            # 세 번째 합성곱 레이어
            layers.Conv2D(64, (3, 3), activation='relu'),

            # Flatten 레이어
            layers.Flatten(),

            # 완전 연결 레이어
            layers.Dense(64, activation='relu'),

            # 출력 레이어
            layers.Dense(10, activation='softmax')
        ])

        # 모델 컴파일
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def load_data(self):
        """CIFAR-10 데이터 로드"""
        print("데이터 로딩 중...")
        cifar10 = keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # 정규화
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # y 데이터를 1차원으로 변환
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        print(f"훈련 데이터: {x_train.shape}, 테스트 데이터: {x_test.shape}")

        return (x_train, y_train), (x_test, y_test)

    def train(self, x_train, y_train, x_test, y_test, epochs=10):
        """모델 학습"""
        print(f"\n모델 학습 시작 (epochs={epochs})...")

        history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            validation_data=(x_test, y_test),
            batch_size=64,
            verbose=1
        )

        return history

    def split_image_to_tiles(self, image, tile_size=32):
        """이미지를 9개의 타일로 분할"""
        tiles = []

        for i in range(3):
            for j in range(3):
                tile = image[i*tile_size:(i+1)*tile_size,
                           j*tile_size:(j+1)*tile_size]
                tiles.append(tile)

        return np.array(tiles)

    def create_captcha_image(self, x_data, y_data, target_class, num_targets=2):
        """CAPTCHA 이미지 생성"""
        captcha_image = np.zeros((96, 96, 3), dtype='float32')
        true_labels = []

        # 타겟 클래스와 기타 클래스 인덱스 찾기
        target_indices = np.where(y_data == target_class)[0]
        other_indices = np.where(y_data != target_class)[0]

        # 타겟 타일 위치 랜덤 선택
        target_positions = np.random.choice(9, num_targets, replace=False)

        tile_idx = 0
        for i in range(3):
            for j in range(3):
                # 타겟 위치인 경우
                if tile_idx in target_positions:
                    img_idx = np.random.choice(target_indices)
                else:
                    img_idx = np.random.choice(other_indices)

                # 이미지 삽입
                captcha_image[i*32:(i+1)*32, j*32:(j+1)*32] = x_data[img_idx]
                true_labels.append(y_data[img_idx])

                tile_idx += 1

        return captcha_image, np.array(true_labels)

    def classify_tiles(self, tiles, target_class):
        """타일 분류 및 타겟 감지"""
        # 모든 타일 예측
        predictions = self.model.predict(tiles, verbose=0)

        # 타겟 클래스가 포함된 타일 찾기
        target_tiles = []

        for i in range(len(tiles)):
            pred_class = np.argmax(predictions[i])

            if pred_class == target_class:
                target_tiles.append(i)

        return predictions, target_tiles

    def visualize_tiles(self, tiles, predictions=None, target_class=None):
        """타일 시각화"""
        plt.figure(figsize=(10, 10))

        for i in range(9):
            plt.subplot(3, 3, i+1)
            plt.imshow(tiles[i])

            if predictions is not None:
                pred_class = np.argmax(predictions[i])
                confidence = predictions[i][pred_class] * 100

                # 타겟 클래스와 일치하면 초록색 테두리
                if target_class is not None and pred_class == target_class:
                    plt.gca().add_patch(plt.Rectangle((0, 0), 31, 31,
                                                      fill=False,
                                                      edgecolor='green',
                                                      linewidth=3))

                # 영어로 표시 (한글 폰트 문제 방지)
                class_name_eng = ['Plane', 'Car', 'Bird', 'Cat', 'Deer',
                                 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
                plt.title(f'{class_name_eng[pred_class]}\n({confidence:.1f}%)',
                         fontsize=10)

            plt.axis('off')

        plt.tight_layout()
        plt.show()


def main():
    """메인 함수"""
    print("=" * 60)
    print("CAPTCHA 타일 인식 시스템 데모")
    print("=" * 60)

    # 시스템 초기화
    recognizer = CaptchaTileRecognizer()

    # 모델 생성
    print("\n1. 모델 생성 중...")
    recognizer.build_model()
    recognizer.model.summary()

    # 데이터 로드
    print("\n2. 데이터 로드 중...")
    (x_train, y_train), (x_test, y_test) = recognizer.load_data()

    # 모델 학습
    print("\n3. 모델 학습 중...")
    print("   (시간이 다소 걸릴 수 있습니다...)")
    history = recognizer.train(x_train, y_train, x_test, y_test, epochs=5)

    # 모델 평가
    print("\n4. 모델 평가 중...")
    test_loss, test_accuracy = recognizer.model.evaluate(x_test, y_test)
    print(f"   테스트 정확도: {test_accuracy*100:.2f}%")

    # CAPTCHA 테스트
    print("\n5. CAPTCHA 시스템 테스트 중...")

    # 자동차 찾기 테스트
    target_class = 1  # 자동차
    print(f"   타겟 객체: {recognizer.class_names[target_class]}")

    # CAPTCHA 이미지 생성
    captcha_img, true_labels = recognizer.create_captcha_image(
        x_test, y_test,
        target_class=target_class,
        num_targets=2
    )

    # 타일 분할
    tiles = recognizer.split_image_to_tiles(captcha_img)

    # 분류
    predictions, detected_tiles = recognizer.classify_tiles(tiles, target_class)

    # 실제 정답
    actual_tiles = np.where(true_labels == target_class)[0].tolist()

    print(f"   감지된 타일: {detected_tiles}")
    print(f"   실제 정답: {actual_tiles}")

    # 정확도 계산
    if set(detected_tiles) == set(actual_tiles):
        print("   결과: 정확!")
    else:
        print("   결과: 불일치")

    # 시각화
    print("\n6. 결과 시각화 중...")
    recognizer.visualize_tiles(tiles, predictions, target_class)

    # 모델 저장
    print("\n7. 모델 저장 중...")
    recognizer.model.save('captcha_tile_model.h5')
    print("   모델이 'captcha_tile_model.h5'로 저장되었습니다.")

    print("\n" + "=" * 60)
    print("데모 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
