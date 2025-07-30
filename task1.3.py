from abc import ABC
from groq import Groq
import os
from pydub import AudioSegment
from pydub.playback import play

class Racer(ABC):

    def __init__(self):
        self.name = ""
        self.tire_health = 100
        self.fuel = 500
        self.offensive_moves = {}
        self.defensive_moves = {}


    def offensive_move(self, move_chosen):
        move_chosen = move_chosen.lstrip().rstrip().lower()
        if move_chosen not in self.offensive_moves:
            print("Invalid Move!!!")
            return False
        else:
            print(f"{self.name} used Tactic: {move_chosen}")
            move_data = self.offensive_moves[move_chosen]
            self.fuel -= move_data["Fuel Cost"]
            return move_data["Impact on Opponent"]


    def defensive_move(self, impact, move_chosen):
        move_chosen = move_chosen.lstrip().rstrip().lower()
        if move_chosen not in self.defensive_moves or self.fuel < 0:
            print("Invalid Move!!!")
            # return False
        else:
            print(f"{self.name} used Tactic: {move_chosen}")
            move_data = self.defensive_moves[move_chosen]
            self.fuel -= move_data["Fuel Cost"]
            self.tire_health -= impact * (1 - move_data["Damage Reduction"])
            # return True

    def print_data(self):
        print(f"{self.name}:")
        print(f"  -  Tire Health: {self.tire_health}.")
        print(f"  -  Fuel: {self.fuel}.")


class Verstappen(Racer):

    def __init__(self):
        super().__init__()
        self.name = "Verstappen"
        self.defensive_moves = {
            "brake late":{"Fuel Cost": 25, "Damage Reduction": 0.3, "Uses": "unlimited"},
            "er5 deployment": {"Fuel Cost": 40, "Damage Reduction": 0.5, "Uses": 3}
        }
        self.offensive_moves = {
            "dr5 boost": {"Fuel Cost": 45, "Impact on Opponent": 12, "Uses": "unlimited"},
            "red bull surge": {"Fuel Cost": 80, "Impact on Opponent": 20, "Uses": "unlimited"},
            "precision turn": {"Fuel Cost": 30, "Impact on Opponent": 8, "Uses": "unlimited"}
                                }


class Hassan(Racer):

    def __init__(self):
        super().__init__()
        self.name = "Hassan"
        self.defensive_moves = {
            "slipstream cut": {"Fuel Cost": 20, "Damage Reduction": 0.4, "Uses": "unlimited"},
            "aggressive block": {"Fuel Cost": 35, "Damage Reduction": 1, "Uses": 2}
        }
        self.offensive_moves = {
            "turbo start": {"Fuel Cost": 50, "Impact on Opponent": 10, "Uses": "unlimited"},
            "mercedes charge": {"Fuel Cost": 90, "Impact on Opponent": 22, "Uses": "unlimited"},
            "corner mastery": {"Fuel Cost": 25, "Impact on Opponent": 7, "Uses": "unlimited"}
        }


def voice_control():
    silence = AudioSegment.silent(duration=1000) # 1 second of silence
    audio_file_path = "sample.wav"
    silence.export(audio_file_path, format="wav")

    return audio_file_path


def get_audio_input():
    # Call the Groq STT API
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    audio_file_path = "sample.wav"

    # Perform the Speech-to-Text transcription
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(audio_file_path, audio_file.read()),
                model="whisper-large-v3-turbo",
                language="en"
            )

        return transcription.text

    except FileNotFoundError:
        print(f"Error: Audio file not found at {audio_file_path}")
    except Exception as e:
        print(f"An error occurred during transcription: {e}")


def main():
    hassan = Hassan()
    verstappen = Verstappen()
    rounds = 1
    players = [hassan, verstappen]
    index = 0

    while True:

        print("_______________________________________________________")
        print(f"Round: {rounds}")
        print("_______________________________________________________")

        offense_move = str(input(f"\n{players[index].name}'s Offense Turn: "))
        off_player = players[index]

        if offense_move.lstrip().rstrip().lower() == "audio":
            voice_control()
            offense_move = get_audio_input()

        if index == 1:
            index = -1

        defense_move = str(input(f"\n{players[index + 1].name}'s Defense Turn: "))
        def_player = players[index + 1]

        if defense_move.lstrip().rstrip().lower() == "audio":
            voice_control()
            defense_move = get_audio_input()

        def_player.defensive_move(off_player.offensive_move(offense_move), defense_move)

        off_player.print_data()
        def_player.print_data()
        print("_______________________________________________________\n")


        rounds += 1
        index += 1
        if hassan.tire_health <= 0:
            print("Hassan lost!!!")
            break
        elif verstappen.tire_health <= 0:
            print("Verstappen lost!!!")
            break


if __name__=="__main__":
    main()