# Используем официальный образ Python (например, 3.8-slim)
FROM python:3.8-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем файл зависимостей в контейнер
COPY requirements.txt .

# Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip && pip install -r requirements.txt

# Копируем весь исходный код проекта в рабочую директорию контейнера
COPY . .
ENV PYTHONPATH=/app
# Открываем порт, на котором будет работать Gradio (по умолчанию 7860)
EXPOSE 7860

# Запускаем приложение; убедитесь, что в main.py указан server_name для корректного запуска в контейнере.
CMD ["python", "-m", "app.main"]
