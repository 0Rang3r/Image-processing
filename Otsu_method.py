def otsu_method(image):
    # Получаем размеры изображения
    height = len(image)
    width = len(image[0])

    # Конвертируем цветное изображение в градации серого
    gray_img = []
    for i in range(height):
        row = []
        for j in range(width):
            R, G, B = image[i][j]  # Извлекаем значения RGB
            # Рассчитываем значение серого по формуле яркости
            gray_pixel = 0.299 * R + 0.587 * G + 0.114 * B
            row.append(gray_pixel)
        gray_img.append(row)

    # Инициализируем гистограмму с 256 нулями
    hist = [0] * 256

    # Вычисляем гистограмму градаций серого
    for i in range(height):
        for j in range(width):
            pixel = int(gray_img[i][j])  # Преобразуем значение пикселя в целое число
            hist[pixel] += 1  # Увеличиваем счетчик для этого значения пикселя

    total_pixels = height * width  # Общее количество пикселей в изображении

    # Инициализируем переменные для метода Отсу
    sum_total = 0  # Сумма произведений интенсивностей пикселей на их количество
    for t in range(256):
        sum_total += t * hist[t]  # Общая накопленная сумма по всему изображению

    weight_background = 0  # Вес фонового класса
    sum_background = 0  # Сумма интенсивностей для фонового класса
    max_between_class_variance = 0  # Максимальная межклассовая дисперсия
    best_t = 0  # Оптимальное значение порога

    # Проходим по всем возможным значениям порога, чтобы найти максимальную межклассовую дисперсию
    for t in range(256):
        weight_background += hist[t]  # Обновляем вес фона
        if weight_background == 0:
            continue  # Пока нет фоновых пикселей, продолжаем

        weight_foreground = total_pixels - weight_background  # Обновляем вес переднего плана
        if weight_foreground == 0:
            break  # Все пиксели относятся к фону

        sum_background += t * hist[t]  # Обновляем сумму фона
        mean_background = sum_background / weight_background  # Средняя интенсивность фона
        mean_foreground = (sum_total - sum_background) / weight_foreground  # Средняя интенсивность переднего плана

        # Вычисляем межклассовую дисперсию
        between_class_variance = ( weight_background * weight_foreground * (mean_background - mean_foreground) ** 2)

        # Обновляем оптимальный порог, если найдено новое максимальное значение дисперсии
        if between_class_variance > max_between_class_variance:
            max_between_class_variance = between_class_variance
            best_t = t

    # Возвращаем оптимальный порог и градации серого изображения
    return best_t, gray_img


def BinaryImage(image):
    # Получаем оптимальный порог и градации серого изображения с помощью метода Отсу
    best_t, gray_img = otsu_method(image)
    height = len(gray_img)
    width = len(gray_img[0])
    bin_img = []

    # Применяем порог для бинаризации изображения
    for i in range(height):
        row = []
        for j in range(width):
            # Устанавливаем пиксель в белый цвет, если значение выше порога, иначе в черный
            if gray_img[i][j] > best_t:
                pixel = 255
            else:
                pixel = 0
            row.append(pixel)
        bin_img.append(row)

    # Возвращаем бинаризованное изображение
    return bin_img
