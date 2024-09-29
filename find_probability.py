def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def calculate_probability(n):
    # Set B: first n odd numbers
    B = [2 * i + 1 for i in range(n)]
    
    # Set A: prime numbers up to the maximum of set B
    A = [i for i in range(2, max(B) + 1) if is_prime(i)]
    
    # Calculate P(A | B)
    P_A_given_B = len(A) / len(B) if B else 0
    
    return P_A_given_B, len(A), max(B)

# Example usage
n = 1000
probability, prime_count, upper_limit_B = calculate_probability(n)
print(f"P(A | B) for the first {n} odd numbers is: {probability}")
print(f"Number of prime numbers: {prime_count}")
print(f"Upper limit of set B: {upper_limit_B}")

