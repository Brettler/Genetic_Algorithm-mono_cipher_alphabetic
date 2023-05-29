import argparse
import sys

def mon_Sub(plainAlp, cipherAlp, message, b_blocks_of_five, b_crypt_mode, b_keep_char):
    inputAlphabet = ''
    outputAlphabet = ''
    if(b_crypt_mode):  # encrypt
        inputAlphabet = plainAlp
        outputAlphabet = cipherAlp
    else:              # decrypt
        inputAlphabet = cipherAlp
        outputAlphabet = plainAlp

    decrypted_message = ""
    # iterate throw message
    for character in message:
        # if the character is in the plaintextalphabet
        if character in inputAlphabet:
            new_char_index = inputAlphabet.index(character)
            new_char = outputAlphabet[new_char_index]
        # if character is not in plaintextalphabet
        elif(not b_keep_char):
            continue
        else:
            if(b_blocks_of_five and b_crypt_mode):
                # if b_blocks_of_five is true skip spaces in message
                if(character != " "):
                    new_char = character
                else:
                    continue
            else:
                # if b_blocks_of_five is false then take all chars, even spaces
                new_char = character
        decrypted_message += new_char

        if(b_blocks_of_five and b_crypt_mode):
            # if b_blocks_of_five is true, then there are no spaces in decrypted_message, so if length is % 5 == 0 append space
            if(len(decrypted_message.replace(" ", "")) % 5 == 0):
                decrypted_message = decrypted_message + " "

    return decrypted_message

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(""-ca", "--cipherAlphabet", help="set ciphertext alphabet", type=str")
    parser.add_argument("-pa", "--plainAlphabet", help="set plaintext alphabet", type=str)
    parser.add_argument("-e", "--encrypt", dest="crypt_mode", action="store_true")
    parser.add_argument("-d", "--decrypt", dest="crypt_mode", action="store_false")
    parser.add_argument("-bf", "--blocks-of-five", dest="blocks_of_five", action="store_true")
    parser.add_argument("-m", "--message", help="text to decrypt / encrypt", type=str, required=True)
    parser.add_argument("-ka", "--keep-non-alp", help="keep non-alphabet characters", dest="keep_char", action="store_true")

    if len(sys.argv) == 1:
        sys.exit(1)

    args = parser.parse_args()

    print(mon_Sub(args.plainAlphabet, args.cipherAlphabet,
            args.message, args.blocks_of_five, args.crypt_mode, args.keep_char))