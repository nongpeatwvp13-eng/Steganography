FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Hugging Face บังคับใช้พอร์ต 7860
EXPOSE 7860

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "--timeout", "120", "Backend.app:app"]