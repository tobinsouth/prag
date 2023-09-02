import numpy as np

MOD = 2**32  # Define a modulus for operations in a finite field
SCALE_FACTOR = 2**32  # Scale factor to convert floats to integers

def float_to_int(vector: np.array(float)) -> np.array(np.int64):
    """Converts a vector of floats to integers by scaling. Note that embedding vectors are in (0,1), so we both shift into the positive domain and scale."""
    return ((vector + 1)*SCALE_FACTOR).astype(np.int64)

def int_to_float(vector: np.array(np.int64)) -> np.array(float):
    """Reverse operation of float_to_int."""
    return (vector.astype(float) / SCALE_FACTOR) - 1

def secret_sharing(vector: np.array(np.int64)):
    """
    Split the vector into two shares. Only works with positive integers.
    """
    random_share = np.random.randint(0, MOD, size=vector.shape)
    secret_share = (vector - random_share) % MOD
    return random_share, secret_share

def reconstruct(share1, share2):
    """
    Reconstruct the original vector from the two shares.
    """
    return (share1 + share2) % MOD


if __name__ == "__main__":
    # Example:
    float_vector = np.random.rand(10)-0.5  # Generate a random float vector

    # Convert float vector to int vector
    int_vector = float_to_int(float_vector)

    # Secret Sharing
    share1, share2 = secret_sharing(int_vector)
    print(f"Share1: {share1}")
    print(f"Share2: {share2}")

    # Reconstruction
    reconstructed_int_vector = reconstruct(share1, share2)
    reconstructed_float_vector = int_to_float(reconstructed_int_vector)
    print(f"Reconstructed Float Vector: {reconstructed_float_vector}")
    print(float_vector-reconstructed_float_vector)



    # Test the reconstruction error for different scale factors
    results = []
    for i in range(5, 50):
        SCALE_FACTOR = 2**i
        float_vector = np.random.rand(1000)*2-1
        int_vector = float_to_int(float_vector)
        share1, share2 = secret_sharing(int_vector)
        reconstructed_int_vector = reconstruct(share1, share2)
        reconstructed_float_vector = int_to_float(reconstructed_int_vector)
        res = np.mean((reconstructed_float_vector - float_vector)**2)   
        results.append(res)
    import matplotlib.pyplot as plt
    plt.plot(range(5, 50), results)
    plt.xlabel("Float to int Scale Factor")
    plt.ylabel("Reconstruction Error")
    plt.yscale('log')
    plt.show()

