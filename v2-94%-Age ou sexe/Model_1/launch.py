import pyaudio 
import wave, matplotlib.pyplot as plt 
import joblib, librosa, numpy as np
from librosa import feature
from keras import models
from IPython.display import Image, display
from os.path import join
import pygame
from pygame.locals import  *



# load model and scaler
model = models.load_model("best_models/network")
scaler = joblib.load("best_models/scaler")


labels = [ 'fifties', 'fourties', 'seventies', 'sixties', 'teens', 'thirties', 'twenties']

gender_dict = {"male": -1, "female": +1, "other": 0}

def app_feature_extraction(path, gender, sampling_rate = 16000):
    features = list()
    audio, _ = librosa.load(path, sr=sampling_rate)
    
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sampling_rate))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sampling_rate))
    spectral_zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    features.append(gender)
    features.append(spectral_centroid)
    features.append(spectral_bandwidth)
    features.append(spectral_rolloff)
    #features.append(spectral_contrast)
    #features.append(spectral_zero_crossing_rate)
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate)
    for el in mfcc:
        features.append(np.mean(el))
    
    return np.asarray(features, dtype=float)


def my_voice_prediction(path, gender, model, scaler, test_number):
    features = app_feature_extraction(path, gender)
    gender = features[0]
    features = scaler.transform(features.reshape(1, -1))  # reshape because we have a single sample
    features = features[0]   # beacause the shape is (1, 24), but we want (24, ) as shape
    features[0] = gender     # in this way the gender will be always +1, 0 or -1
    prediction = model.predict(np.expand_dims(features, axis=0))
    #plt.plot(labels[:-1], prediction[0])
    #plt.show()
    print("L'age prédit par le model {} est : {}".format(test_number, labels[np.argmax(prediction)]))
    return labels[np.argmax(prediction)], prediction


def main():
    n_test = 0

    pygame.init()

    window = pygame.display.set_mode((700,600))
    
    screen = pygame.display.get_surface()
    pygame.display.set_caption('Voice Age - Capgemini')
    
    police = pygame.font.SysFont("arial", 74)
    police_1 = pygame.font.SysFont("arial", 27)
    police_title = pygame.font.SysFont("arial",50)
        
    chunk = 1024  
  
    sample_format = pyaudio.paInt16   
    chanels = 1
  
    smpl_rt = 48000  
    seconds = 4
    image_texte_1 = police_1.render('Bonjour, je teste ma voix pour l\'AIE capgemini.', 1, (255,0,0))
    image_texte_2 = police_1.render('Appuyer sur ESPACE pour lancer l\'enregistrement.', 1, (255,0,0))
    image_texte_3 = police_title.render('Demo voice age - AIE Capgemini',1, (255,0,0))
    image_texte_recording = police_1.render("Enregistrement en cours...", 1, (255,0,0))

    continuer = True
    
    window.fill((255,255,255))
    pygame.display.flip()

    state = 'P'
    title = True

    while continuer:

        if state == 'P':
            pygame.display.update()
            screen.blit(image_texte_2, (130,550))
            if title:
                screen.blit(image_texte_3, (60,270))
                title = False
        
        if state == 'R':
            window.fill((255,255,255))
            ### Enregistrement de la voix
            
            filename = "enregistrement.wav"
      
            pa = pyaudio.PyAudio()   
      
            stream = pa.open(format=sample_format, channels=chanels,
                         rate=smpl_rt, input=True,  
                         frames_per_buffer=chunk)
      
            print('Vous pouvez parler (4 seconds) ')
            print('Enregistrement en cours...')

            record = True

            if record:
                screen.blit(image_texte_1, (130,300))
                screen.blit(image_texte_recording, (10,10))
                pygame.display.flip()
                record = False
      
            frames = []   
      
            for i in range(0, int(smpl_rt / chunk * seconds)): 
                data = stream.read(chunk) 
                frames.append(data) 
      
            stream.stop_stream() 
            stream.close() 
      
            pa.terminate()
            screen.fill((255,255,255))
            pygame.display.update()
      
            print('Estimation de l\'age ... ') 
            gender = 0
            sf = wave.open(filename, 'wb') 
            sf.setnchannels(chanels) 
            sf.setsampwidth(pa.get_sample_size(sample_format)) 
            sf.setframerate(smpl_rt) 
            sf.writeframes(b''.join(frames)) 
            sf.close()
            n_test+=1
            
            pr, probs = my_voice_prediction("enregistrement.wav", gender, model, scaler, test_number=0)
            #display(Image(filename = ))
            
            print(probs)
            #labels = [ 'fifties', 'fourties', 'seventies', 'sixties', 'teens', 'thirties', 'twenties']
            labels_ord = {labels[4] : probs[0][4],
                          labels[6] : probs[0][6],
                          labels[5] : probs[0][5],
                          labels[1] : probs[0][1],
                          labels[0] : probs[0][0],
                          labels[3] : probs[0][3],
                          labels[2] : probs[0][2]
                          }
            plt.bar(labels_ord.keys(), labels_ord.values())


            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.title('Age probabilities')

            plt.savefig('results.png')
            plt.close()
            src_img = join("images",pr+".png")
            age = pr

            image = pygame.image.load(src_img)
            image = image.convert()
            image = pygame.transform.scale(image, (200,200))

            image_1 = pygame.image.load('results.png')
            image_1 = image_1.convert()
            image_1 = pygame.transform.scale(image_1, (350,350))

        
            image_texte = police.render(age, 1, (255,0,0))
            screen.blit(image,(400,240))
            screen.blit(image_1,(0,200))
            screen.blit(image_texte, (240,100))

            state = 'P'
            #re = input("Voulez vous arrêter ? (y/n) : ")
            #if re == 'y':
                #print('Merci !')
                #break
            
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.type == pygame.K_ESCAPE:
                    continuer = False
                    pygame.quit()
                    pygame.display.flip()
                if event.key == pygame.K_SPACE:
                    state = 'R'
        pygame.display.flip()
    pygame.quit()

        

if __name__== "__main__":
    main()
