
#Biblioteka do UI
import gradio as gr
import torch
import tempfile
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any

# Import our encapsulated inference engine
from src.predict import GalaxyPredictor

# Webowy UI do Klasyfikatora Galaktyk
class GalaxyUI:

    def __init__(self, predictor: GalaxyPredictor) -> None:

       # Wstrzykiwanie zależności - UI otrzyma z zewnątrz gotowy
       # załadowany do pamięci model przewidujący
        self.predictor: GalaxyPredictor = predictor
        
        # Wywołujemy metodę która kompiluje "klocki" interfejsu
        # Zmienna self.app przechowuje gotowy serwer webowy w stanie uśpienia.
        self.app: gr.Blocks = self._build_interface()

    def _predict_wrapper(self, image: np.ndarray) -> Dict[str, float]:
        
      
        if image is None:
            return {"Brak obrazu": 0.0}  

        # Gradio przesyła surową macierz pikseli. Zmieniamy ją na obraz PIL
        pil_image = Image.fromarray(image)

        # Zapisywanie tymczasowe: nasz model oczekuje ścieżki do pliku,
        # więc tworzymy w systemie ukryty, tymczasowy plik .jpg
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
           temp_path = Path(temp_file.name)
           pil_image.save(temp_path)

        try:
            # Wysyłamy ścieżkę do silnika i odbieramy surowe wyniki
            raw_predictions = self.predictor.predict(temp_path)
        finally:
            # Usuwamy plik tymczasowy nawet gdy model wyrzuci błąd 
            if temp_path.exists():
                temp_path.unlink()  
           
       # Zmieniamy techniczne numery klas na zrozumiałe nazwy
        class_mapping ={
            0: "Gładka (Smooth)",
            1: "Spiralna (Spiral)",
            2: "Gwiazda/Artefakt (Star/Artifact)",
        }

        # Tworzymy nowy słownnik, zastępując cyfry (0,1,2) tekstem
        mapped_predictions = {}
        for class_idx in range(3):
            prob = raw_predictions.get(class_idx, 0.0)
            human_name =  class_mapping.get(class_idx, f"Klasa {class_idx}")
            mapped_predictions[human_name] = prob

        # zwracamy słownik a Gradio automatycznie wyrenderuje na jego podstawie
        # wykres słupkowy w przeglądarce
        return mapped_predictions

    def _build_interface(self) -> gr.Blocks:

       # gr.Blocks to puste "płótno", na którym układamy komponenty.
        with gr.Blocks(title="Universe Cataloger") as interface:

            # Nagłówki w języku Markdown
            gr.Markdown("# 🌌 Universe Cataloger: Galaxy Morphology Classifier")
            gr.Markdown("Upload a raw telescope image to predict its morphological structure.")

            with gr.Row():
                with gr.Column():
                    #Okienko drag&drop dla zdjęć galaktyk
                    image_input = gr.Image(label="Telescope Image", type="numpy")
                    submit_btn = gr.Button("Analyze Morphology")
                
                with gr.Column():
                    # gr.Label automatycznie interpretuje przekazany słownik 
                    # jako dane do estetycznego wykresu słupkowego.
                    label_output = gr.Label(num_top_classes=5, label="Predicted Probabilities")

            # Spinamy kliknięcia z kodem
            submit_btn.click(
                fn=self._predict_wrapper,
                inputs=[image_input],
                outputs=[label_output]
            )

        return interface

    def launch(self, **kwargs: Any) -> None:
       
        print("Launching Universe Cataloger UI...")

        # Uruchamiamy lokalny serwer webowy
        # **kwargs pozwala przekazać dowolną liczbę dodatkowych parametrów
        # np. share=True (tworzy publiczny link)
        self.app.launch(**kwargs)


def main() -> None:
    
    # 1. Konfiguracja środowiska
    weights_path = Path("models/best_galaxy_cnn.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 37

    print(f"Booting Inference Engine on {device}...")
    
    # 2.Inicjalizacja backendu
    # Tworzymy instancję naszego silnika predykcyjnego. Ten proces potrwa chwilę, 
    # ponieważ ładuje z dysku gigabajty wag neuronowych do pamięci RAM/VRAM.
    predictor = GalaxyPredictor(
        weights_path=weights_path, 
        device=device, 
        num_classes=num_classes
    )
    
    # 3. Wstrzykiwanie zależności i uruchomienie frontendu
    ui = GalaxyUI(predictor=predictor)
    ui.launch(share=False)


if __name__ == "__main__":
    main()