import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import random
import time
import streamlit as st

model = load_model("model.h5")
label = np.load("labels.npy")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Use streamlit to create the web app
st.title("Facial Expression Game")

# OpenCV VideoCapture object


update_time = time.time() + 1
corrected_timer = time.time()
corrected = False
end_time = -1
correct_count = 0
current_expression = random.choice(['happy', 'sad', 'angry', 'surprise'])
pred = ""
start_time = time.time()

# Streamlit widget for displaying the video feed
video_placeholder = st.empty()

# Streamlit checkbox to stop the video feed
stop_checkbox = st.checkbox("Stop")

# Streamlit checkbox to reset the timer and correct count
restart_checkbox = st.button("Restart")

def randomsec():
	global current_expression
	global update_time
	global corrected
	# if (time.time() >= update_time) or (corrected == True):
	if (corrected == True):
		current_expression = random.choice(['happy','sad','angry','surprise'])
		# update_time = time.time() + 1
		corrected = False
		
cap = cv2.VideoCapture(1) 

video_width = 1500

while True:
    
    lst = []

    _, frm = cap.read()

    frm = cv2.flip(frm, 1)

    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    if time.time() > start_time:
        end_time += 1
        start_time = time.time() + 1

    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        lst = np.array(lst).reshape(1, -1)

        pred = label[np.argmax(model.predict(lst))]

        cv2.putText(frm, "=" + pred, (620, 650), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

    cv2.putText(frm, current_expression, (470, 650), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
    cv2.putText(frm, str(end_time), (50, 50), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

    if (current_expression in pred) & (corrected == False) & (time.time() > corrected_timer):
        corrected = True
        corrected_timer = time.time()
        correct_count += 1
        # st.write(f"Correct Count: {correct_count}")

    cv2.putText(frm, "" + str(correct_count), (550, 700), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

    video_placeholder.image(frm, channels="BGR", use_column_width=True, width=video_width)
    
    randomsec()
    
    if end_time==45:
       cv2.waitKey(-1)
       

    if stop_checkbox:
        # st.write("Stopping...")
        cap.release()
        break

    if restart_checkbox:
        # st.write("Resetting...")
        end_time = 0
        correct_count = 0
        # reset_checkbox = st.checkbox("Restart")
       
    # if cv2.waitKey(1) == 27:
    #     cv2.destroyAllWindows()
    #     cap.release()
    #     break
    
    # if cv2.waitKey(1) == ord('p'):
    #     cv2.waitKey(-1) 
        
    # if cv2.waitKey(1) == ord('r'):
    #     end_time=0
    #     correct_count=0

    # Streamlit button to stop the video feed
    # if st.button("Stop"):
    #     st.write("Stopping...")
    #     break

    # # Streamlit button to reset the timer and correct count
    # if st.button("Reset"):
    #     st.write("Resetting...")
    #     end_time = 0
    #     correct_count = 0

# Release resources
cap.release()
cv2.destroyAllWindows()
