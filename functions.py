import cv2 as cv
from itertools import product

def collect_data(func) -> None:

    def wrapper():
        # Create a MediaPipe Holistic object for hand tracking and landmark extraction
        with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
            # Loop through each action, sequence, and frame to record data
            for action, sequence, frame in product(signs, range(sequences), range(frames)):
                # If it is the first frame of a sequence, wait for the spacebar key press to start recording
                if frame == 0: 
                    while True:
                        if keyboard.is_pressed(' '):
                            break
                        _, image = cap.read()

                        results = image_process(image, holistic)
                        draw_landmarks(image, results)

                        #cv.putText(image, 'Recording data for the "{}". Sequence number {}.'.format(action, sequence),
                        #            (20,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)
                        #cv.putText(image, 'Pause.', (20,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                        #cv2.putText(image, 'Press "Space" when you are ready.', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                        cv.imshow('Camera', image)
                        cv.waitKey(1)
                        
                        # Check if the 'Camera' window was closed and break the loop
                        if cv.getWindowProperty('Camera',cv.WND_PROP_VISIBLE) < 1:
                            break
                else:
                    # For subsequent frames, directly read the image from the camera
                    _, image = cap.read()
                    # Process the image and extract hand landmarks using the MediaPipe Holistic pipeline
                    results = image_process(image, holistic)
                    # Draw the hand landmarks on the image
                    draw_landmarks(image, results)

                    # Display text on the image indicating the action and sequence number being recorded
                    #cv.putText(image, 'Recroding data for the "{}". Sequence number {}.'.format(action, sequence),
                    #            (20,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)
                    cv.imshow('Camera', image)
                    cv.waitKey(1)

                # Check if the 'Camera' window was closed and break the loop
                if cv.getWindowProperty('Camera',cv.WND_PROP_VISIBLE) < 1:
                    break

                # Extract the landmarks from both hands and save them in arrays
                keypoints = keypoint_extraction(results)
                frame_path = os.path.join(PATH, action, str(sequence), str(frame))
                np.save(frame_path, keypoints)

            # Release the camera and close any remaining windows
            cap.release()
            cv.destroyAllWindows()