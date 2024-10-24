import unittest
from Otsu_method import BinaryImage
import numpy as np
import cv2


class TestOtsuThresholding(unittest.TestCase):
    def setUp(self):
        # Создаем тестовые изображения для проверки
        # Контрастное изображение с черными и белыми пикселями
        self.contrast_image = np.array([[[255, 255, 255], [0, 0, 0]],
                                        [[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        # Шумовое изображение с случайными значениями пикселей
        self.noisy_image = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        # Пустое изображение, все пиксели установлены в черный цвет
        self.blank_image = np.zeros((50, 50, 3), dtype=np.uint8)
        # Изображение, где все пиксели имеют среднее серое значение
        self.grey_image = np.full((50, 50, 3), 127, dtype=np.uint8)
        # Градиентное изображение от черного к белому
        self.gradient_image = np.linspace(0, 255, 100*100).reshape((100, 100)).astype(np.uint8)
        self.gradient_image = cv2.merge([self.gradient_image, self.gradient_image, self.gradient_image])
        # Темное изображение с низкими значениями интенсивности
        self.dark_image = np.full((50, 50, 3), 30, dtype=np.uint8)
        # Светлое изображение с высокими значениями интенсивности
        self.bright_image = np.full((50, 50, 3), 220, dtype=np.uint8)
        # Изображение с случайными значениями пикселей от 0 до 255
        self.random_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        # Горизонтальный градиент от черного к белому
        self.horizontal_gradient = np.tile(np.linspace(0, 255, 50).astype(np.uint8), (50, 1))
        self.horizontal_gradient = cv2.merge([self.horizontal_gradient, self.horizontal_gradient, self.horizontal_gradient])
        # Вертикальный градиент от черного к белому
        self.vertical_gradient = np.tile(np.linspace(0, 255, 50).astype(np.uint8), (50, 1)).T
        self.vertical_gradient = cv2.merge([self.vertical_gradient, self.vertical_gradient, self.vertical_gradient])

    def test_contrast_image(self):
        # Тестируем пороговую обработку Отсу на контрастном изображении
        binary_image = BinaryImage(self.contrast_image)
        binary_image = np.array(binary_image, dtype=np.uint8)  # Преобразуем в NumPy массив
        # Ожидаемый результат: чередующиеся черные и белые пиксели
        expected_output = np.array([[255, 0], [0, 255]], dtype=np.uint8)
        self.assertTrue(np.array_equal(binary_image, expected_output))

    def test_noisy_image(self):
        # Тестируем пороговую обработку Отсу на шумовом изображении
        binary_image = BinaryImage(self.noisy_image)
        binary_image = np.array(binary_image, dtype=np.uint8)
        # Проверяем, что выходное бинарное изображение имеет правильную форму
        self.assertTrue(binary_image.shape == (100, 100))

    def test_blank_image(self):
        # Тестируем пороговую обработку Отсу на пустом изображении (все пиксели черные)
        binary_image = BinaryImage(self.blank_image)
        binary_image = np.array(binary_image, dtype=np.uint8)
        # Ожидаемый результат: все пиксели черные или белые
        self.assertTrue(np.all(binary_image == 0) or np.all(binary_image == 255))

    def test_grey_image(self):
        # Тестируем пороговую обработку Отсу на однородном сером изображении
        binary_image = BinaryImage(self.grey_image)
        binary_image = np.array(binary_image, dtype=np.uint8)
        # Бинарное изображение может быть полностью черным или белым, в зависимости от порога
        self.assertTrue(np.all(binary_image == 0) or np.all(binary_image == 255))

    def test_gradient_image(self):
        # Тестируем пороговую обработку Отсу на градиентном изображении
        binary_image = BinaryImage(self.gradient_image)
        binary_image = np.array(binary_image, dtype=np.uint8)
        # Проверяем, что выходное бинарное изображение имеет правильную форму
        self.assertTrue(binary_image.shape == (100, 100))

    def test_dark_image(self):
        # Тестируем пороговую обработку Отсу на темном изображении
        binary_image = BinaryImage(self.dark_image)
        binary_image = np.array(binary_image, dtype=np.uint8)
        # Ожидаемый результат: все пиксели черные или белые
        self.assertTrue(np.all(binary_image == 0) or np.all(binary_image == 255))

    def test_bright_image(self):
        # Тестируем пороговую обработку Отсу на светлом изображении
        binary_image = BinaryImage(self.bright_image)
        binary_image = np.array(binary_image, dtype=np.uint8)
        # Ожидаемый результат: все пиксели черные или белые
        self.assertTrue(np.all(binary_image == 0) or np.all(binary_image == 255))

    def test_random_image(self):
        # Тестируем пороговую обработку Отсу на изображении со случайными пикселями
        binary_image = BinaryImage(self.random_image)
        binary_image = np.array(binary_image, dtype=np.uint8)
        # Проверяем, что выходное бинарное изображение имеет правильную форму
        self.assertTrue(binary_image.shape == (50, 50))

    def test_horizontal_gradient(self):
        # Тестируем пороговую обработку Отсу на горизонтальном градиенте
        binary_image = BinaryImage(self.horizontal_gradient)
        binary_image = np.array(binary_image, dtype=np.uint8)
        # Проверяем, что выходное бинарное изображение имеет правильную форму
        self.assertTrue(binary_image.shape == (50, 50))

    def test_vertical_gradient(self):
        # Тестируем пороговую обработку Отсу на вертикальном градиенте
        binary_image = BinaryImage(self.vertical_gradient)
        binary_image = np.array(binary_image, dtype=np.uint8)
        # Проверяем, что выходное бинарное изображение имеет правильную форму
        self.assertTrue(binary_image.shape == (50, 50))

if __name__ == '__main__':
    # Запускаем единичные тесты
    unittest.main()
