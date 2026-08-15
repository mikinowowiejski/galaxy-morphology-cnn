# 🌌 Universe Cataloger: Deep Learning Galaxy Morphology

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)

Kompleksowy system uczenia maszynowego do automatycznej klasyfikacji morfologii galaktyk na podstawie zdjęć z teleskopu (zbiór danych Galaxy Zoo). Projekt obejmuje pełen potok MLOps: od niestandardowego ładowania i augmentacji danych, przez dwufazowy Transfer Learning, aż po serwowanie modelu w interfejsie graficznym.

---

## 🚀 Główne cechy architektury

*   **Model:** Pre-trenowany **ResNet18** (Transfer Learning) dostosowany do regresji miękkich prawdopodobieństw (37 klas morfologicznych).
*   **Strategia Dwufazowa (Two-Phase Training):** 
    *   *Faza 1:* Zamrożony rdzeń (backbone), agresywny trening nowej głowicy regresyjnej.
    *   *Faza 2:* Odmrożenie rdzenia, fine-tuning z drastycznie obniżonym współczynnikiem uczenia w celu wyłapania astronomicznych detali.
*   **Optymalizacja pętli uczącej:** Zaimplementowano mechanizmy **Early Stopping** oraz **ReduceLROnPlateau** zabezpieczające przed przeuczeniem (Overfitting).
*   **Solidny Data Pipeline:** Własna implementacja `torch.utils.data.Dataset` połączona z dynamiczną augmentacją obrazu (odwrócenia, rotacje 180°, normalizacja ImageNet) i stratyfikowanym podziałem na zbiory (Train/Val).
*   **Interfejs Użytkownika:** Interaktywna aplikacja webowa zbudowana w oparciu o bibliotekę **Gradio** do weryfikacji modelu w czasie rzeczywistym.

---

## 🛠 Wyzwania Inżynieryjne i Rozwiązania

Podczas tworzenia projektu napotkałem kilka zaawansowanych problemów związanych z sieciami konwolucyjnymi:
1. **Mode Collapse (Zapaść do średniej):** Początkowa, prosta sieć CNN (zbudowana od zera) faworyzowała uśrednianie wyników ze względu na skomplikowaną strukturę wizualną galaktyk. Rozwiązanie: Wdrożenie głębszej sieci z mechanizmem uwagi na detale poprzez **Transfer Learning**.
2. **Training-Serving Skew:** Różnice w normalizacji danych pomiędzy fazą uczenia a środowiskiem produkcyjnym. Rozwiązanie: Ujednolicenie potoku przetwarzania obrazów (ImageNet Mean/Std Normalization) w klasie `GalaxyPredictor`.
3. **Double Sigmoid Anomaly:** Model trenowany z użyciem `MSELoss` na danych ułamkowych z definicji zwracał gotowe prawdopodobieństwa. Aplikacja błędnie aplikowała na nie drugą funkcję sigmoid, kompresując wyniki. Rozwiązanie: Usunięcie redundancji i implementacja rygorystycznego blokowania wartości (`torch.clamp`) w inferencji.

---

## 📂 Struktura Projektu

Projekt został zorganizowany w oparciu o zasady Clean Code i programowania obiektowego (OOP):

```text
katalogowanie_wszechswiata/
│
├── data/
│   └── raw/                # Surowe pliki .jpg oraz plik .csv z etykietami (ignorowane przez Git)
├── models/                 # Zapisane wagi modelu (.pth)
├── src/                    # Rdzeń inżynieryjny:
│   ├── data_loader.py      # Transformacje, Datasets, DataLoaders
│   ├── eda_utils.py        # Audyt i wczytywanie danych
│   ├── model_cnn.py        # Architektura ResNet18 (modyfikacja głowicy)
│   ├── train.py            # Obiektowy silnik trenujący (GalaxyTrainer)
│   └── predict.py          # Logika inferencji modelu dla UI
├── ui/
│   └── app.py              # Aplikacja Gradio
├── main.py                 # Skrypt orkiestrujący trening (Two-Phase Pipeline)
├── requirements.txt        # Zależności projektu
└── README.md

---

