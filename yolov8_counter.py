import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import time
from datetime import datetime
from ultralytics import YOLO

class PersonCounterYOLOv8:
    def __init__(self, video_path, output_path=None, conf_threshold=0.5, save_video=True):

        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.save_video = save_video

        # caminho do output
        if output_path:
            self.output_path = output_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = f"output_{timestamp}.mp4"

        # carrega o modelo da yolo
        self.model = self.load_model()

        # abre o vídeo pra capturar pessoas
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

        # configs
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps,
                (self.width, self.height)
            )

        # counter e rastreador de variáveis
        self.total_persons = 0
        self.current_persons = 0
        self.tracker = {}
        self.tracked_ids = set()
        self.next_id = 1
        self.frame_count = 0

        # definição da linha de contagem (que por enquanto está no meio da tela)
        self.line_position = self.height // 2

        # definição da região acima e abaixo da linha de cont
        self.upper_region = set()  
        self.lower_region = set()  

        self.up_count = 0
        self.down_count = 0

    def load_model(self):
        try:
            # yolov8
            model = YOLO('yolov8s.pt')  # tem as seguintes opcões: 's' ,'n', 'm', 'l', 'x', o x é mais lento mas tbm é bem preciso
            return model
        except Exception as e:
            print(f"Erro ao carregar o modelo YOLOv8: {e}")
            raise ImportError(
                "Não foi possível carregar o modelo YOLOv8. "
                "Por favor, instale-o com: pip install ultralytics"
            )

    def process_frame(self, frame):

        # roda a yolo
        results = self.model(frame, classes=0)

        # rastreando pessoas
        self.current_persons = 0
        current_tracked = set()

        line_color = (0, 255, 255)  # linha amrela (da pra mudar a cor
        cv2.line(frame, (0, self.line_position), (self.width, self.line_position),
                line_color, 2)

        # abel q fica na pessoa
        cv2.putText(frame, "Linha de Contagem", (self.width//2 - 80, self.line_position - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                # coordenadas do box
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                conf = float(box.conf[0])

                # verificação do nível de confiança 
                if conf < self.conf_threshold:
                    continue

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                tracked_id = None
                min_dist = float('inf')

                for id, pos in self.tracker.items():
                    dist = np.sqrt((center_x - pos[0])**2 + (center_y - pos[1])**2)
                    if dist < min_dist and dist < 100:  
                        min_dist = dist
                        tracked_id = id


                if tracked_id is None:
                    tracked_id = self.next_id
                    self.next_id += 1
                    self.tracked_ids.add(tracked_id)

                self.tracker[tracked_id] = (center_x, center_y)
                current_tracked.add(tracked_id)
                self.current_persons += 1

                # contabilizando pessoas de acordo com a linha
                is_in_upper = center_y < self.line_position
                is_in_lower = center_y > self.line_position

                if is_in_upper and tracked_id not in self.upper_region:
                    self.upper_region.add(tracked_id)
           
                    if tracked_id in self.lower_region:
                        self.total_persons += 1
                        self.up_count += 1
                        self.lower_region.remove(tracked_id)

                if is_in_lower and tracked_id not in self.lower_region:
                    self.lower_region.add(tracked_id)
     
                    if tracked_id in self.upper_region:
                        self.total_persons += 1
                        self.down_count += 1
                        self.upper_region.remove(tracked_id)

                # box verde (fora da loja
                color = (0, 255, 0) if is_in_upper else (255, 0, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                label = f"ID: {tracked_id}, {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        ids_to_remove = set(self.tracker.keys()) - current_tracked
        for id in ids_to_remove:
            del self.tracker[id]

        # mostrando os dados na tela do vídeo
        cv2.putText(frame, f"Pessoas na tela: {self.current_persons}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Total de passagens: {self.total_persons}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def run(self):
        print(f"Processando vídeo: {self.video_path}")
        print(f"Output: {self.output_path if self.save_video else 'Sem gravação'}")
        print("Pressione 'q' para sair")

        start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Processa o frame
            processed_frame = self.process_frame(frame)

            # Dmostra o frame
            cv2.imshow('Monitoramento de Pessoas (YOLOv8)', processed_frame)

            if self.save_video:
                self.writer.write(processed_frame)

            # fps
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time

            cv2.putText(processed_frame, f"FPS: {fps:.2f}",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # digitar 'q' se quiser sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        if self.save_video:
            self.writer.release()
        cv2.destroyAllWindows()

        # mostrando dados finais
        print(f"\nResultados finais:")
        print(f"Total de passagens detectadas: {self.total_persons}")
        print(f"Total de frames processados: {frame_count}")
        print(f"Tempo total de processamento: {elapsed_time:.2f} segundos")
        print(f"FPS médio: {fps:.2f}")

        return {
            "total_persons": self.total_persons,
            "processing_time": elapsed_time,
            "fps": fps,
            "up_count": self.up_count,
            "down_count": self.down_count
        }

def main():

    parser = argparse.ArgumentParser(description='Monitor e conta pessoas em um vídeo usando YOLOv8')
    parser.add_argument('--video', type=str, required=True, help='Caminho para o vídeo de entrada')
    parser.add_argument('--output', type=str, default=None, help='Caminho para salvar o vídeo de saída')
    parser.add_argument('--conf', type=float, default=0.5, help='Limite de confiança para detecção (0-1)')
    parser.add_argument('--save', action='store_true', help='Salvar o vídeo de saída')

    args = parser.parse_args()

    # verifica se o vídeo existe
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Erro: O arquivo de vídeo não existe: {args.video}")
        return

    # cria e roda script de contabilização
    counter = PersonCounterYOLOv8(
        video_path=str(video_path),
        output_path=args.output,
        conf_threshold=args.conf,
        save_video=args.save
    )

    results = counter.run()

    print("\nProcessamento concluído!")
    print(f"Total de pessoas contadas: {results['total_persons']}")

if __name__ == "__main__":
    main()
