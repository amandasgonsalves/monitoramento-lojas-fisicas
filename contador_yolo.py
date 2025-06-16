import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import time
from datetime import datetime
from ultralytics import YOLO
import math
from enum import Enum


class PersonState(Enum):
    EXTERNAL = "external"  # Pessoa completamente fora da loja
    ENTERING = "entering"  # Pessoa parcialmente entrando
    INTERNAL = "internal"  # Pessoa completamente dentro da loja
    EXITING = "exiting"    # Pessoa parcialmente saindo


class PersonCounterYOLOv8:
    def __init__(self, video_path, output_path=None, conf_threshold=0.5,
                 save_video=True, angle_deg=30):

        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.save_video = save_video
        self.angle_deg = angle_deg

        # Caminho do vídeo de saída
        if output_path:
            self.output_path = output_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = f"output_{timestamp}.mp4"

        # Carrega modelo YOLO
        self.model = self.load_model()

        # Abre o vídeo
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (self.width, self.height))

        # Inicialização de variáveis de contagem
        self.current_persons = 0
        self.tracker = {}
        self.tracked_ids = set()
        self.next_id = 1

        # Novo sistema de estados
        self.person_states = {}  # {person_id: PersonState}
        self.entry_count = 0     # Pessoas que entraram na loja
        self.exit_count = 0      # Pessoas que saíram da loja

        # Define pontos da linha inclinada
        self.set_diagonal_line()

    def set_diagonal_line(self):
        theta = math.radians(self.angle_deg)
        length = math.hypot(self.width, self.height)
        cx, cy = self.width // 2, self.height // 2 -30

        dx = int((length / 2) * math.cos(theta))
        dy = int((length / 2) * math.sin(theta))

        self.line_p1 = (cx - dx, cy - dy)
        self.line_p2 = (cx + dx, cy + dy)
        
        # Definir zona de segurança (buffer) para evitar oscilações
        self.buffer_distance = 30  # pixels de buffer

    def load_model(self):
        try:
            model = YOLO('yolov8s.pt')
            return model
        except Exception as e:
            print(f"Erro ao carregar o modelo YOLOv8: {e}")
            raise ImportError(
                "Não foi possível carregar o modelo YOLOv8. "
                "Por favor, instale-o com: pip install ultralytics"
            )

    def classify_bbox_position(self, x1, y1, x2, y2):
        """
        Classifica onde a bounding box está em relação à linha de contagem
        Retorna: 'external', 'partial', 'internal'
        """
        # Calcula a distância de todos os pontos da bbox para a linha
        Ax, Ay = self.line_p1
        Bx, By = self.line_p2
        
        # Pontos da bounding box
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        
        # Calcula em que lado da linha cada canto está
        sides = []
        for px, py in corners:
            side = (Bx - Ax) * (py - Ay) - (By - Ay) * (px - Ax)
            sides.append(side)
        
        # Verifica se todos os pontos estão do mesmo lado
        # INVERTENDO A LÓGICA: valores negativos = fora, valores positivos = dentro
        all_external = all(s < -self.buffer_distance for s in sides)  # Todos abaixo da linha (fora)
        all_internal = all(s > self.buffer_distance for s in sides)   # Todos acima da linha (dentro)
        
        if all_external:
            return 'external'
        elif all_internal:
            return 'internal'
        else:
            return 'partial'

    def update_person_state(self, person_id, bbox_position):
        """
        Atualiza o estado de uma pessoa e conta entradas/saídas
        REGRA: Só conta quem começou FORA da loja
        """
        # Se é uma pessoa nova (primeira detecção)
        if person_id not in self.person_states:
            # Define estado inicial baseado na posição atual
            if bbox_position == 'external':
                initial_state = PersonState.EXTERNAL
            elif bbox_position == 'partial':
                initial_state = PersonState.ENTERING  # Assumir que estava entrando
            else:  # bbox_position == 'internal'
                initial_state = PersonState.INTERNAL  # NÃO CONTA - já estava dentro
            
            self.person_states[person_id] = initial_state
            return initial_state
        
        # Para pessoas já rastreadas, aplicar lógica normal
        current_state = self.person_states[person_id]
        new_state = current_state
        
        if current_state == PersonState.EXTERNAL:
            if bbox_position == 'partial':
                new_state = PersonState.ENTERING
            elif bbox_position == 'internal':
                new_state = PersonState.INTERNAL
                self.entry_count += 1  # ENTRADA VÁLIDA: veio de fora
        
        elif current_state == PersonState.ENTERING:
            if bbox_position == 'internal':
                new_state = PersonState.INTERNAL
                self.entry_count += 1  # ENTRADA VÁLIDA: completou entrada
            elif bbox_position == 'external':
                new_state = PersonState.EXTERNAL  # Desistiu de entrar
        
        elif current_state == PersonState.INTERNAL:
            if bbox_position == 'partial':
                new_state = PersonState.EXITING
            elif bbox_position == 'external':
                new_state = PersonState.EXTERNAL
                self.exit_count += 1  # SAÍDA VÁLIDA
        
        elif current_state == PersonState.EXITING:
            if bbox_position == 'external':
                new_state = PersonState.EXTERNAL
                self.exit_count += 1  # SAÍDA VÁLIDA: completou saída
            elif bbox_position == 'internal':
                new_state = PersonState.INTERNAL  # Voltou para dentro
        
        self.person_states[person_id] = new_state
        return new_state

    def process_frame(self, frame):
        results = self.model(frame, classes=0)

        self.current_persons = 0
        current_tracked = set()

        cv2.line(frame, self.line_p1, self.line_p2, (0, 255, 255), 2)
        cv2.putText(frame, "Linha de Contagem", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                conf = float(box.conf[0])

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

                # Classifica posição da bounding box completa
                bbox_position = self.classify_bbox_position(x1, y1, x2, y2)
                
                # Atualiza estado da pessoa
                person_state = self.update_person_state(tracked_id, bbox_position)
                
                # Define cor baseada no estado
                color_map = {
                    PersonState.EXTERNAL: (0, 0, 255),    # Vermelho - fora
                    PersonState.ENTERING: (0, 255, 255),  # Amarelo - entrando
                    PersonState.INTERNAL: (0, 255, 0),    # Verde - dentro
                    PersonState.EXITING: (255, 0, 255)    # Magenta - saindo
                }
                color = color_map.get(person_state, (255, 255, 255))

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"ID: {tracked_id} [{person_state.value}] {conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Remove IDs antigos
        ids_to_remove = set(self.tracker.keys()) - current_tracked
        for id in ids_to_remove:
            del self.tracker[id]
            # Limpar também os estados
            if id in self.person_states:
                del self.person_states[id]

        # Informações na tela
        cv2.putText(frame, f"Pessoas na tela: {self.current_persons}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Entradas: {self.entry_count}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Saidas: {self.exit_count}",
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Saldo: {self.entry_count - self.exit_count}",
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

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

            processed_frame = self.process_frame(frame)

            cv2.imshow('Monitoramento de Pessoas (YOLOv8)', processed_frame)

            if self.save_video:
                self.writer.write(processed_frame)

            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time

            cv2.putText(processed_frame, f"FPS: {fps:.2f}",
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        if self.save_video:
            self.writer.release()
        cv2.destroyAllWindows()

        print(f"\nResultados finais:")
        print(f"Total de entradas: {self.entry_count}")
        print(f"Total de saídas: {self.exit_count}")
        print(f"Pessoas ainda dentro: {self.entry_count - self.exit_count}")
        print(f"Total de frames processados: {frame_count}")
        print(f"Tempo total de processamento: {elapsed_time:.2f} segundos")
        print(f"FPS médio: {fps:.2f}")

        return {
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "current_inside": self.entry_count - self.exit_count,
            "processing_time": elapsed_time,
            "fps": fps
        }


def main():
    parser = argparse.ArgumentParser(description='Monitor e conta pessoas em um vídeo usando YOLOv8')
    parser.add_argument('--video', type=str, required=True, help='Caminho para o vídeo de entrada')
    parser.add_argument('--output', type=str, default=None, help='Caminho para salvar o vídeo de saída')
    parser.add_argument('--conf', type=float, default=0.5, help='Limite de confiança para detecção (0-1)')
    parser.add_argument('--save', action='store_true', help='Salvar o vídeo de saída')
    parser.add_argument('--angle', type=float, default=30, help='Ângulo da linha de contagem em graus')

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Erro: O arquivo de vídeo não existe: {args.video}")
        return

    counter = PersonCounterYOLOv8(
        video_path=str(video_path),
        output_path=args.output,
        conf_threshold=args.conf,
        save_video=args.save,
        angle_deg=args.angle
    )

    results = counter.run()
    print("\nProcessamento concluído!")
    print(f"Total de entradas: {results['entry_count']}")
    print(f"Total de saídas: {results['exit_count']}")
    print(f"Pessoas ainda dentro da loja: {results['current_inside']}")


if __name__ == "__main__":
    main()