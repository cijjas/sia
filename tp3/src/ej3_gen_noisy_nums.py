import json
import numpy as np
from utils.noise_factory import NoiseFactory, NoiseType

CONFIG_PATH = "../config/ej3_noise.json"

# we get the noise type and the output file path from the config file
def main():
    with open(CONFIG_PATH, 'r') as file:
        data = json.load(file)
    noise = data.get('noise', None)
    output_noise = data.get('output_noise', None)

    if noise is None or output_noise is None:
        raise ValueError("Invalid config file")
    
    if noise == 'gaussian':
        NoiseFactory.generate_noisy_numbers(NoiseType.GAUSSIAN, output_noise, data)
    elif noise == 'salt_and_pepper':
        NoiseFactory.generate_noisy_numbers(NoiseType.SALT_AND_PEPPER, output_noise, data)
    else:
        raise ValueError("Invalid noise type")
    
    print(f"Noisy numbers saved to {output_noise}")

    # now we have a 70x5 matrix in the output file which has noise for the numbers to be applied
    # i also have a 70x5 matrix of the same numbers but without noise in the file data.get('nums_path', None)
    # i will apply a xor to the file, saving it in the file data.get('output_xor', None)
    # remember there are only 0's and 1's in the txt's and that the xor is applied to the whole matrix, not to each number separately

    output_xor = data.get('output_xor', None)
    nums_path = data.get('nums_path', None)

    if output_xor is None or nums_path is None:
        raise ValueError("Invalid config file")
    
    with open(output_noise, 'r') as file:
        noisy_numbers = np.loadtxt(file)

    with open(nums_path, 'r') as file:
        numbers = np.loadtxt(file)

    xor = np.logical_xor(noisy_numbers, numbers)

    np.savetxt(output_xor, xor, fmt='%d')

    print(f"XOR applied to noisy numbers saved to {output_xor}")


if __name__ == "__main__":
    main()
