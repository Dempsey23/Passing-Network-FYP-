import LEI_VS_ARS
import LEI_VS_CITY
import LEI_VS_STO
import TOT_VS_NEWCASTLE
import TOT_VS_STOKE
import TOT_VS_UNITED
from LEI_VS_ARS import A_LEI as LEI_ARS_ADJ
from LEI_VS_STO import A_LEI as LEI_STO_ADJ
from LEI_VS_TOT import A_LEI as LEI_TOT_ADJ
from LEI_VS_CITY import A_LEI as LEI_CITY_ADJ
from TOT_VS_UNITED import A_TOT as TOT_UNI_ADJ
from TOT_VS_STOKE import A_TOT as TOT_STO_ADJ
from TOT_VS_NEWCASTLE import A_TOT as TOT_NEW_ADJ
from TOT_VS_LEICESTER import A_TOT as TOT_LEI_ADJ
import networkx as nx
import numpy as np

sourceFile=open("Similarity_Between_Matches_brute_forceLEI&TOT.txt",'w')
min_value = min(np.min(LEI_ARS_ADJ), np.min(LEI_STO_ADJ),np.min(LEI_CITY_ADJ),np.min(TOT_UNI_ADJ), np.min(TOT_STO_ADJ),np.min(TOT_NEW_ADJ),np.min(TOT_LEI_ADJ),np.min(LEI_TOT_ADJ))
max_value = max(np.max(LEI_ARS_ADJ), np.max(LEI_STO_ADJ),np.max(LEI_CITY_ADJ),np.max(TOT_UNI_ADJ), np.max(TOT_STO_ADJ),np.max(TOT_NEW_ADJ),np.min(TOT_LEI_ADJ),np.min(LEI_TOT_ADJ))

# Normalize both matrices to the range [0, 1]
normalized_LEI_ARS = (LEI_ARS_ADJ - min_value) / (max_value - min_value)
normalized_LEI_STO = (LEI_STO_ADJ - min_value) / (max_value - min_value)
normalized_LEI_CITY = (LEI_CITY_ADJ - min_value) / (max_value - min_value)

normalized_TOT_UNI = (TOT_UNI_ADJ - min_value) / (max_value - min_value)
normalized_TOT_NEW = (TOT_NEW_ADJ - min_value) / (max_value - min_value)
normalized_TOT_STO = (TOT_STO_ADJ - min_value) / (max_value - min_value)

normalized_TOT_LEI = (TOT_LEI_ADJ - min_value) / (max_value - min_value)
normalized_LEI_TOT = (LEI_TOT_ADJ - min_value) / (max_value - min_value)

def spectral_distance(G1, G2):
    # Get the number of nodes in each graph
    n1 = len(G1.nodes())
    n2 = len(G2.nodes())

    # Compute the adjacency matrices of the graphs
    A1 = nx.linalg.graphmatrix.adjacency_matrix(G1).toarray()
    A2 = nx.linalg.graphmatrix.adjacency_matrix(G2).toarray()


    # Compute the difference between adjacency matrices
    diff = A1 - A2

    # Compute the spectral norm (maximum singular value) of the difference
    spectral_norm = np.linalg.norm(diff, ord='fro')  # Frobenius norm

    return spectral_norm

def sum_of_absolute_differences_by_row(matrixA, matrixB):
    # Check if matrices have the same dimensions
    if len(matrixA) != len(matrixB) or len(matrixA[0]) != len(matrixB[0]):
        raise ValueError("Matrices must have the same dimensions")

    total_sum = 0
    # Iterate over rows
    for i in range(len(matrixA)):
        row_sum = 0  # Initialize sum for the current row
        # Iterate over columns
        for j in range(len(matrixA[0])):
            # Calculate the absolute difference for the current cell
            row_sum += abs(matrixA[i][j] - matrixB[i][j])

        # Add row sum to total sum
        total_sum += row_sum
        print(f"Sum after row {i + 1}: {total_sum}")
        print(f"    Sum of this row {i + 1}: {row_sum}")
    return total_sum
#Leicester
print('Similarity LEI passing VS STO & VS ARS (WIN/LOSS)\n---------------\n',sum_of_absolute_differences_by_row(normalized_LEI_STO,normalized_LEI_ARS),file=sourceFile)
print(sum_of_absolute_differences_by_row(normalized_LEI_STO,normalized_LEI_ARS))

print('Similarity LEI passing VS CITY & VS ARS(DRAW/LOSS) \n---------------\n',sum_of_absolute_differences_by_row(normalized_LEI_CITY,normalized_LEI_ARS),file=sourceFile)
print(sum_of_absolute_differences_by_row(normalized_LEI_CITY,normalized_LEI_ARS))

print('Similarity LEI passing VS CITY & VS STO(DRAW/WIN) \n---------------\n',sum_of_absolute_differences_by_row(normalized_LEI_CITY,normalized_LEI_STO),file=sourceFile)
print(sum_of_absolute_differences_by_row(normalized_LEI_CITY,normalized_LEI_STO))

#Tottenham
print('Similarity TOT passing VS UNITED & VS NEWCASTLE (WIN/LOSS)\n---------------\n',sum_of_absolute_differences_by_row(normalized_TOT_UNI,normalized_TOT_NEW),file=sourceFile)
print(sum_of_absolute_differences_by_row(normalized_TOT_UNI,normalized_TOT_NEW))

print('Similarity TOT passing VS STOKE & VS NEWCASTLE (DRAW/LOSS) \n---------------\n',sum_of_absolute_differences_by_row(normalized_TOT_STO,normalized_TOT_NEW),file=sourceFile)
print(sum_of_absolute_differences_by_row(normalized_TOT_STO,normalized_TOT_NEW))

print('Similarity TOT passing VS STOKE & VS UNITED (DRAW/WIN) \n---------------\n',sum_of_absolute_differences_by_row(normalized_TOT_STO,normalized_TOT_UNI),file=sourceFile)
print(sum_of_absolute_differences_by_row(normalized_TOT_STO,normalized_TOT_UNI))

#Tottenham & Leicester (Same result similarity)
print('-----------------------',file=sourceFile)
print('Similarity of TOT passing VS UNITED & LEI passing vs STOKE (WIN\WIN)\n---------------\n',sum_of_absolute_differences_by_row(normalized_TOT_UNI,normalized_LEI_STO),file=sourceFile)

print('Similarity of TOT passing VS NEWCASTLE & LEI passing vs ARSENAL (LOSS\LOSS)\n---------------\n',sum_of_absolute_differences_by_row(normalized_TOT_NEW,normalized_LEI_ARS),file=sourceFile)

print('Similarity of TOT passing VS STOKE & LEI passing vs CITY (DRAW\DRAW)\n---------------\n',sum_of_absolute_differences_by_row(normalized_TOT_STO,normalized_LEI_CITY),file=sourceFile)


#Tottenham & Leicester (Different result Similarity)
print('Similarity of TOT passing VS UNITED & LEI passing vs ARSENAL (WIN(TOT)\LOSS(LEI)\n---------------\n',sum_of_absolute_differences_by_row(normalized_TOT_UNI, normalized_LEI_ARS), file=sourceFile)

print('Similarity of TOT passing VS NEWCASTLE & LEI passing vs ARSENAL (DRAW(TOT)\LOSS(LEI)\n---------------\n',sum_of_absolute_differences_by_row(normalized_TOT_STO, normalized_LEI_ARS), file=sourceFile)



print('Similarity of TOT passing VS STOKE & LEI passing vs CITY (DRAW(TOT)\WIN(LEI)\n---------------\n',sum_of_absolute_differences_by_row(normalized_TOT_STO, normalized_LEI_STO), file=sourceFile)

print('Similarity of TOT passing VS UNITED & LEI passing vs STOKE (LOSS(TOT)\WIN(LEI)\n---------------\n',sum_of_absolute_differences_by_row(normalized_TOT_NEW, normalized_LEI_STO), file=sourceFile)



print('Similarity of TOT passing VS NEWCASTLE & LEI passing vs ARSENAL (LOSS(TOT)\DRAW(LEI)\n---------------\n',sum_of_absolute_differences_by_row(normalized_TOT_NEW, normalized_LEI_CITY), file=sourceFile)

print('Similarity of TOT passing VS STOKE & LEI passing vs CITY (WIN(TOT)\DRAW(LEI)\n---------------\n',sum_of_absolute_differences_by_row(normalized_TOT_UNI, normalized_LEI_CITY), file=sourceFile)

#Spectral Distance
print('\n-------Spectral Distance----------',file=sourceFile)
print('\nSpectral Distance TOT VS STO & VS UNITED\n',spectral_distance(TOT_VS_STOKE.G_TOT,TOT_VS_UNITED.G_TOT),file=sourceFile)
print('Spectral Distance TOT VS NEWCASTLE & VS UNITED\n',spectral_distance(TOT_VS_NEWCASTLE.G_TOT,TOT_VS_UNITED.G_TOT),file=sourceFile)
print('Spectral Distance TOT VS NEWCASTLE & VS STOKE\n',spectral_distance(TOT_VS_NEWCASTLE.G_TOT,TOT_VS_STOKE.G_TOT),file=sourceFile)

print('Spectral Distance LEI VS STO & VS ARS (WIN/LOSS)\n---------------\n',spectral_distance(LEI_VS_STO.G_LEI,LEI_VS_ARS.G_LEI),file=sourceFile)
print('Spectral Distance LEI VS CITY & VS ARS(DRAW/LOSS) \n---------------\n',spectral_distance(LEI_VS_CITY.G_LEI,LEI_VS_ARS.G_LEI),file=sourceFile)
print('Spectral Distance LEI VS CITY & VS STO(DRAW/WIN) \n---------------\n',spectral_distance(LEI_VS_CITY.G_LEI,LEI_VS_STO.G_LEI),file=sourceFile)