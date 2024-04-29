def terme_suite(n):
    
    U = []
    for i in range(n+1):
        if i == 0:
            U.append(0) 
        else:
            U.append(U[i-1]+50)
    print(f"le {n}-ieme element de cette suite vaut: {U[n]}")
    # if n == 0:
    #     return 0
    # else:
    #     return terme_suite(n-1) + 50

n = int(input("Entrez la valeur de n : "))
resultat = terme_suite(n)
# print(f"Le {n}-ième terme de la suite est : {resultat}")

def pgcd(a, b):
    while b != 0:
        a, b = b, a % b
        print(f"a = {a} b = {b}")
    return a

# a = int(input("Entrez le premier nombre a : "))
# b = int(input("Entrez le deuxième nombre b : "))

# res = pgcd(a, b)
# print(f"Le PGCD de {a} et {b} est : {res}")

def convert_to_decimal(num, base):
    if base == 2:
        return int(num, 2)
    elif base == 8:
        return int(num, 8)
    elif base == 10:
        return int(num)
    elif base == 16:
        return int(num, 16)
    else:
        return "Base non supportée"

def convert_from_decimal(num, base):
    if base == 2:
        return bin(num)[2:]
    elif base == 8:
        return oct(num)[2:]
    elif base == 10:
        return str(num)
    elif base == 16:
        return hex(num)[2:]
    else:
        return "Base non supportée"

# num = input("Entrez le nombre : ")
# base = int(input("Entrez la base du nombre (2 pour binaire, 8 pour octal, 10 pour décimal, 16 pour hexadécimal) : "))

# decimal_num = convert_to_decimal(num, base)

# print("En binaire :", convert_from_decimal(decimal_num, 2))
# print("En octal :", convert_from_decimal(decimal_num, 8))
# print("En décimal :", convert_from_decimal(decimal_num, 10))
# print("En hexadécimal :", convert_from_decimal(decimal_num, 16))
