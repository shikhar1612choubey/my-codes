import pyttsx3
import speech_recognition as sr
import webbrowser
import datetime
import pyjokes
import wikipedia
import os
import time
import pyaudio


def speechtext():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            print("recognizing...")
            data = recognizer.recognize_google(audio)
            return data
        except sr.UnknownValueError:
            print("Not Understanding")


def speak(audio):
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    engine.setProperty('voice', voices[0].id)
    rate = engine.getProperty('rate')
    engine.setProperty('rate', 135)
    engine.say(audio)
    engine.runAndWait()


def wishme():
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        speak("Good Morning")
    elif 12 <= hour < 18:
        speak("Good Afternoon")
    else:
        speak("good evening")

    speak("I am Jarvis sir. Please tell me how may I help you")


if __name__ == '__main__':
    wishme()
    while True:
        data1 = speechtext().lower()
        if "your name" in data1:
            name = "my name is Jarvis"
            speak(name)

        elif "old are you" in data1:
            age = "My age is infinite and I will never die "
            speak(age)

        elif "time" in data1:
            time = (datetime.datetime.now().strftime("%I%M%p"))
            speak(time)

        elif 'wikipedia' in data1:
            speak('searching Wikipedia...')
            data1 = data1.replace("wikipedia", "")
            results = wikipedia.summary(data1, sentences=2)
            speak("According to Wikipedia")
            print(results)
            speak(results)

        elif 'youtube' in data1:
            speak('opening youtube...')
            webbrowser.open('https://www.youtube.com/')

        elif 'gmail' in data1:
            speak('opening gmail...')
            webbrowser.open('https://mail.google.com/mail/u/0/#inbox')

        elif "joke" in data1:
            joke_1 = pyjokes.get_joke(language="en", category="neutral")
            print(joke_1)
            speak(joke_1)

        elif 'play video' in data1:
            address = "E:\movies"
            listvid = os.listdir(address)
            print(listvid)
            os.startfile(os.path.join(address, listvid[9]))

        elif "exit" in data1:
            speak("thankyou Shikhar")
            break
        time.sleep(5)
