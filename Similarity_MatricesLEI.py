from LEI_VS_ARS import A_LEI as LEI_ARS_ADJ
from LEI_VS_STO import A_LEI as LEI_STO_ADJ
from LEI_VS_TOT import A_LEI as LEI_TOT_ADJ
from LEI_VS_CITY import A_LEI as LEI_CITY_ADJ
import numpy as np

sourceFile=open("Similarity_Between_Matches_brute_forceLEI.txt",'w')
min_value = min(np.min(LEI_ARS_ADJ), np.min(LEI_STO_ADJ),np.min(LEI_CITY_ADJ))
max_value = max(np.max(LEI_ARS_ADJ), np.max(LEI_STO_ADJ),np.max(LEI_CITY_ADJ))

# Normalize both matrices to the range [0, 1]
normalized_LEI_ARS = (LEI_ARS_ADJ - min_value) / (max_value - min_value)
normalized_LEI_STO = (LEI_STO_ADJ - min_value) / (max_value - min_value)
normalized_LEI_CITY = (LEI_CITY_ADJ - min_value) / (max_value - min_value)


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

print('Similarity LEI passing VS STO & VS ARS (WIN/LOSS)\n---------------\n',sum_of_absolute_differences_by_row(normalized_LEI_STO,normalized_LEI_ARS),file=sourceFile)
print(sum_of_absolute_differences_by_row(normalized_LEI_STO,normalized_LEI_ARS))

print('Similarity LEI passing VS CITY & VS ARS(DRAW/LOSS) \n---------------\n',sum_of_absolute_differences_by_row(normalized_LEI_CITY,normalized_LEI_ARS),file=sourceFile)
print(sum_of_absolute_differences_by_row(normalized_LEI_CITY,normalized_LEI_ARS))

print('Similarity LEI passing VS CITY & VS STO(DRAW/WIN) \n---------------\n',sum_of_absolute_differences_by_row(normalized_LEI_CITY,normalized_LEI_STO),file=sourceFile)
print(sum_of_absolute_differences_by_row(normalized_LEI_CITY,normalized_LEI_STO))