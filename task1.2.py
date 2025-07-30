class Codec:

    def encode(self, list_of_commands: list[str]) -> str:
        """
        Encodes a list of command strings into a single string.
        Each command is prefixed with its length and separated by a null terminator"\x00.
        Example: ["Push now", "Box, box"] -> "8:Push now\x008:Box, box\x00"
        """
        encoded_string = ""

        #
        if not list_of_commands:
            return encoded_string

        for command in list_of_commands:
            encoded_string += f"8:{command}\x00"

        return encoded_string


    def decode(self, encoded_string) :
        """
        Decodes a single, encoded string back into the original list of command strings and prints it.
        """
        decoded_commands = []

        if not encoded_string:
            return []

        start_flag = False
        end_flag = True
        id = 0

        for char in encoded_string:
            if char == ":":
                start_flag = True
                end_flag = False
                decoded_commands.append("")
                continue
            elif char == "\x00":
                start_flag = False
                end_flag = True
                id += 1
            elif start_flag == True and end_flag == False:
                decoded_commands[id] += char

        return decoded_commands


def main():
    codec = Codec()

    # Scenario 1: Basic F1 commands
    commands1 = [
        "Push now",
        "Box, box",
        "Check temperatures",
        "Fuel save mode activated",
        "Tyre deg very high",
        " "
    ]

    encoded = codec.encode(commands1)
    print(encoded)
    decoded = codec.decode(encoded)
    print(decoded)

if __name__=="__main__":
    main()