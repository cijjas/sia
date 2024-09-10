#include <stdio.h>
#include <math.h>
#include <string.h>
#include <limits.h>

#define NUM_FIELDS 5
#define HEIGHT_START 1.3
#define HEIGHT_END 2.01
#define HEIGHT_STEP 0.01
#define MIN_POINTS 100
#define MAX_POINTS 200

// gcc - O3 - o breaker.out breaker.c - lm

// Precompute tanh lookuprtable for values from 0 to 200
double tanh_lookup[201];

// Function prototypes
void precompute_tanh_lookup();
double eve(const char *character, int strength, int dexterity, int intelligence, int vigor, int constitution, double height);
void write_best_distribution_to_csv(const char *filename, const char *character, double best_value, int *best_distribution, double height, int total_points);
void write_csv_header(const char *filename);

int main()
{

    precompute_tanh_lookup();
    // printf("warrior : <%f>\n", eve("warrior", 50, 50, 0, 0, 0, 1.92));
    // printf("archer : <%f>\n", eve("archer", 50, 50, 0, 0, 0, 1.92));
    const char *classes[] = {"warrior", "archer", "guardian", "mage"};
    int num_classes = sizeof(classes) / sizeof(classes[0]);
    const char *filename = "best_values.csv";
    write_csv_header(filename);

    for (int total_points = MIN_POINTS; total_points <= MAX_POINTS; ++total_points)
    {
        for (int c = 0; c < num_classes; ++c)
        {
            const char *character = classes[c];
            double best_value = -INFINITY;
            int best_distribution[NUM_FIELDS] = {0};
            double best_height = HEIGHT_START;

            for (double height = HEIGHT_START; height < HEIGHT_END; height += HEIGHT_STEP)
            {
                for (int a = 0; a <= total_points; ++a)
                {
                    for (int b = 0; b <= total_points - a; ++b)
                    {
                        for (int d = 0; d <= total_points - a - b; ++d)
                        {
                            for (int e = 0; e <= total_points - a - b - d; ++e)
                            {
                                int f = total_points - a - b - d - e; // Calculate the last attribute automatically

                                int distribution[NUM_FIELDS] = {a, b, d, e, f};
                                double result = eve(character, distribution[0], distribution[1], distribution[2], distribution[3], distribution[4], height);
                                if (result > best_value)
                                {
                                    // printf("%s,%d,%d,%d,%d,%d,%d,%f,%f\n", character, total_points, distribution[0], distribution[1], distribution[2], distribution[3], distribution[4], height, result);
                                    best_value = result;
                                    best_height = height;
                                    memcpy(best_distribution, distribution, sizeof(int) * NUM_FIELDS);
                                }
                            }
                        }
                    }
                }
            }
            write_best_distribution_to_csv(filename, character, best_value, best_distribution, best_height, total_points);
        }
    }
    return 0;
}

void precompute_tanh_lookup()
{
    for (int i = 0; i <= 200; ++i)
    {
        tanh_lookup[i] = tanh(0.01 * i);
    }
}

double eve(const char *character, int strength, int dexterity, int intelligence, int vigor, int constitution, double height)
{
    // Ensure that indices are within bounds
    if (strength < 0 || strength > 200 || dexterity < 0 || dexterity > 200 || intelligence < 0 || intelligence > 200 || vigor < 0 || vigor > 200 || constitution < 0 || constitution > 200)
    {
        return -INFINITY;
    }

    double total_strength = 100 * tanh_lookup[strength];
    double total_dexterity = tanh_lookup[dexterity];
    double total_intelligence = 0.6 * tanh_lookup[intelligence];
    double total_vigor = tanh_lookup[vigor];
    double total_constitution = 100 * tanh_lookup[constitution];

    double atm = 0.5 - pow(3 * height - 5, 4) + pow(3 * height - 5, 2) + height / 2;
    double dem = 2 + pow(3 * height - 5, 4) - pow(3 * height - 5, 2) - height / 2;

    double attack = (total_dexterity + total_intelligence) * total_strength * atm;
    double defense = (total_vigor + total_intelligence) * total_constitution * dem;

    if (strcmp(character, "warrior") == 0)
    {
        return 0.6 * attack + 0.4 * defense;
    }
    else if (strcmp(character, "archer") == 0)
    {
        return 0.9 * attack + 0.1 * defense;
    }
    else if (strcmp(character, "guardian") == 0)
    {
        return 0.1 * attack + 0.9 * defense;
    }
    else if (strcmp(character, "mage") == 0)
    {
        return 0.8 * attack + 0.2 * defense;
    }

    return -INFINITY;
}

void write_best_distribution_to_csv(const char *filename, const char *character, double best_value, int *best_distribution, double height, int total_points)
{
    FILE *file = fopen(filename, "a");
    if (file != NULL)
    {
        fprintf(file, "%s,%d,%d,%d,%d,%d,%d,%f,%f\n", character, total_points, best_distribution[0], best_distribution[1], best_distribution[2], best_distribution[3], best_distribution[4], height, best_value);
        fclose(file);
    }
}

void write_csv_header(const char *filename)
{
    FILE *file = fopen(filename, "w");
    if (file != NULL)
    {
        fprintf(file, "Class,Total Points,Strength,Dexterity,Intelligence,Vigor,Constitution,Height,Value\n");
        fclose(file);
    }
}