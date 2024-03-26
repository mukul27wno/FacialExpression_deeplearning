import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import cv2
import numpy as np
from keras.models import load_model
import random
import time
import mediapipe as mp

st.init_session_state()

model = load_model("model.h5")
label = np.load("labels.npy")

holistic = mp.solutions.holistic
holis = holistic.Holistic()

update_time = time.time() + 1
corrected_timer = time.time()
corrected = False
end_time = -1
correct_count = 0
current_expression = random.choice(['happy', 'sad', 'angry', 'surprise'])
pred = ""
start_time = time.time()

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.holis = holistic.Holistic()

    def transform(self, frame):
        lst = []

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = self.holis.process(frame_rgb)

        if time.time() > start_time:
            global end_time
            end_time += 1
            start_time = time.time() + 1

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]

            cv2.putText(frame, "=" + pred, (620, 650), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

        cv2.putText(frame, current_expression, (470, 650), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
        cv2.putText(frame, str(end_time), (50, 50), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

        if (current_expression in pred) & (corrected == False) & (time.time() > corrected_timer):
            corrected
            corrected = True
            corrected_timer = time.time()
            global correct_count
            correct_count += 1
            # st.write(f"Correct Count: {correct_count}")

        cv2.putText(frame, "" + str(correct_count), (550, 700), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

        return frame

def main():
    st.title("Facial Expression Game")

    webrtc_ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        async_transform=True,
    )

if __name__ == "__main__":
    main()
