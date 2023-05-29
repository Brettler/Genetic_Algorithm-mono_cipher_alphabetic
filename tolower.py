


with open('enc2.txt', 'r') as f:
    cipher_text = f.read()

cipher_text = cipher_text.lower()

with open('enc2Lower.txt', 'w') as f:
    f.write(cipher_text)

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


