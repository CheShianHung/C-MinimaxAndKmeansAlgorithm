/*	CS6200  Che Shian Hung  4/13/2018
Programming Assignment 3
Purpose: This program implements minimax and kmeans classifying algorithms to classify
		 randomly generated samples, and display the testing results. The samples are 
		 generated from five circle centers. Each circle center is used to generated 
		 10 samples within a given radius. The user can test with different radius
		 in the interactive interface. The user can also test with different numbers
		 of algorithm setting to perform testing through interactive interface. Each
		 program run will generate different set of testing samples. However, we can 
		 test with the sampe data again and agian in the terminal. In this program, 
		 the distance between two points is the similarity measure for both algorithm.
		 For minimax algo., it will find a potential class center in each iteration, 
		 and include the new class center into the result when it passes the test 
		 with the previous class centers, and the testing result and stats are captured
		 by several global variables. For kmeans algo, before the algorithm runs, the
		 user can indicate the number of classes given for kmeans algorithm. Initially,
		 the algorithm will randomly select the same number of samples as initial class 
		 centers, and reclassifying the samples and calculating the class centers again
		 and again until the result converge. The output of the each test result is 
		 format as how we generate data. Since the data is generated for each circle
		 center sequencially, the samples for each circle center are separated from
		 each other in the test result.
*/

#define _USE_MATH_DEFINES

// Import libraries and constants
#include<iostream>
#include<stdlib.h>
#include<time.h>
#include<cmath>
#include<math.h>
#include<string>

#define CENTER_NUM 5				// Number of circle centers
#define DIMENSION 2					// Dimensionality for each sample
#define SAMPLE_SIZE 10				// Number of sample generate for each class
#define MAX_ITERATION 1000			// Iteration limit for training with backpropagation network

using namespace std;

// Linklist node to capture the runtime generated class center in minimax algorithm
struct centerIndexNode {
	int index;
	centerIndexNode* next;
};

const double sphereCenter[CENTER_NUM][DIMENSION] = {{6, -6}, { -6, -6}, {10, 6}, {-10, 6}, {0, 10}};	// Hard coded circle centers
double samples[CENTER_NUM * SAMPLE_SIZE][DIMENSION];				// Sample for all classes

int minimaxClassNum = 0;											// Number of classes after applying minimax
int minimaxResultClass[CENTER_NUM * SAMPLE_SIZE];					// Arrays that capture the classifying result after applying minimax
int* minimaxCenterIndex = NULL;										// Arrays of the classes center index from the samples after applying minimax
double** minimaxClassCenters = NULL;								// 2D arrays of finalized class centers after applying minimax

int kmeansClassNum;													// Number of classes after applying kmeans
int kmeansIterationNum;												// Number of iteration took from previous kmeans
int kmeansResultClass[CENTER_NUM * SAMPLE_SIZE];					// Arrays that capture the classifying result after applying kmeans
double** kmeansClassCenters = NULL;									// 2D arrays of finalized class centers after applying kmeans

void generateAllSamples(int radius);								// Given a radius, generate all samples randomly within each class center
void displayAllSamples();											// Output all samples
void displayMinimaxResult();										// Display testing result after applying minimax
void displayKmeansResult();											// Display testing result after applying kmeans 
void deletePointers();												// Delete global pointers
int userInput(int mode);											// Ask user for input along with different question modes and return an integer answer

void minimax();														// Perfrom minimax algo.
void kmeans(int cNum);												// Perform kmeans algo.
double distance(double* a, double* b);								// Compute the distance between two points
centerIndexNode* appendToIndex(centerIndexNode* list, int index);	// Append a new centerIndexNode to the linklist while running minimax



int main() {
	srand(time(NULL));

	generateAllSamples(userInput(1));						// Generate sample data 
	displayAllSamples();									// Display sample data

	do {
		if (userInput(2) == 1) {							// Ask user to decide which algorithm to run
			minimax();										// Run minimax
			displayMinimaxResult();							// Display minimax result
		}
		else {
			kmeans(userInput(3));							// Run kmeans
			displayKmeansResult();							// Display kmeans result
		}
	} while (userInput(4) == 2);							// Ask user to see if continue testing or exit the program

	deletePointers();										// Delete all pointers
	system("pause");
	return 0;
}

void generateAllSamples(int radius) {
	for (int i = 0; i < CENTER_NUM * SAMPLE_SIZE; i++) {
		double theta = rand() % 6282 / double(1000);
		double r = rand() % (radius * 1000) / double(1000);
		samples[i][0] = sphereCenter[i / SAMPLE_SIZE][0] + r * cos(theta);
		samples[i][1] = sphereCenter[i / SAMPLE_SIZE][1] + r * sin(theta);
	}
}

void displayAllSamples() {
	printf("\ndisplay all samples:\n");
	for (int i = 0; i < CENTER_NUM * SAMPLE_SIZE; i++)
		printf("(%6.2f, %6.2f)\n", samples[i][0], samples[i][1]);
	printf("\n\n");
}

void displayMinimaxResult() {
	printf("\ndisplay minimaxResult:\n\n");
	printf("number of class: %d\n\n", minimaxClassNum);
	printf("number of iteration: %d\n\n", minimaxClassNum - 1);
	for (int i = 0; i < CENTER_NUM * SAMPLE_SIZE; i++) {
		printf("(%6.2f, %6.2f) => class %d\n", samples[i][0], samples[i][1], minimaxResultClass[i]);
		if (i % SAMPLE_SIZE == SAMPLE_SIZE - 1) printf("\n");
	}
	printf("\n\n");
	printf("display class centers:\n\n");
	for (int i = 0; i < minimaxClassNum; i++) {
		printf("class %d: (%6.2f, %6.2f), original index: %d\n", i + 1, minimaxClassCenters[i][0], minimaxClassCenters[i][1], minimaxCenterIndex[i]);
	}
	printf("\n\n");
}

void displayKmeansResult() {
	printf("display kmeansResult:\n\n");
	printf("number of class: %d\n\n", kmeansClassNum);
	printf("number of iteration: %d\n\n", kmeansIterationNum);
	for (int i = 0; i < CENTER_NUM * SAMPLE_SIZE; i++) {
		printf(" (%6.2f, %6.2f) => class %d\n", samples[i][0], samples[i][1], kmeansResultClass[i]);
		if (i % SAMPLE_SIZE == SAMPLE_SIZE - 1) printf("\n");
	}
	printf("\n\n");
	printf("display class centers:\n\n");
	for (int i = 0; i < kmeansClassNum; i++) {
		printf("class %d: (%6.2f, %6.2f)\n", i + 1, kmeansClassCenters[i][0], kmeansClassCenters[i][1]);
	}
	printf("\n\n");
}

void minimax() {
	// Initialize variables
	int cNum = 1;
	double centerDistanceTotal = 0;
	bool done = false;
	bool isCenter[CENTER_NUM * SAMPLE_SIZE];
	for (int i = 0; i < CENTER_NUM * SAMPLE_SIZE; i++) {
		isCenter[i] = false;
		minimaxResultClass[i] = 0;
	}
	isCenter[0] = true;
	
	// Contruct centerIndexNode linklist to capture runtime generated class centers
	centerIndexNode* indexHead = new centerIndexNode();
	indexHead->index = 0;
	indexHead->next = NULL;

	// Each minimax iteration
	while (!done && cNum < MAX_ITERATION){
		double maxResult[2] = { 0, -1 };		// Capture the result for each iteration

		// Compute and compare distance between each sample and classCenters
		for (int i = 0; i < CENTER_NUM * SAMPLE_SIZE; i++) {		
			if (!isCenter[i]) {
				centerIndexNode* h = indexHead;
				double minResult[2];
				minResult[0] = distance(samples[i], samples[h->index]);
				minResult[1] = (double) i;
				while (h->next) {
					h = h->next;
					double d = distance(samples[i], samples[h->index]);
					if (d < minResult[0]) {
						minResult[0] = d;
						minResult[1] = i;
					}
				}
				h = NULL;
				if (minResult[0] > maxResult[0]) {
					maxResult[0] = minResult[0];
					maxResult[1] = minResult[1];
				}
			}
		}

		// Check if there is a new cluster center
		if (cNum != 1 && maxResult[0] <= centerDistanceTotal / (cNum * (cNum - 1) / 2) / 2) {
			done = true;
		}
		
		// If yes, add the new cluster center and update stats
		if (!done) {
			centerIndexNode* h = indexHead;
			for (int i = 0; i < cNum; i++) {
				centerDistanceTotal += distance(samples[h->index], samples[(int) maxResult[1]]);
				h = h->next;
			}
			h = NULL;
			cNum++;
			indexHead = appendToIndex(indexHead, (int) maxResult[1]);
			isCenter[(int) maxResult[1]] = true;
		}
	}

	// Put cNum linked list into an array and build the array for calculating cluster centers
	if (minimaxCenterIndex) delete[] minimaxCenterIndex;
	minimaxCenterIndex = new int[cNum];
	double** classTotalPosition = new double*[cNum];
	centerIndexNode* indexTracer = indexHead;
	for (int i = 0; i < cNum; i++) {
		classTotalPosition[i] = new double[2];
		classTotalPosition[i][0] = 0;
		classTotalPosition[i][1] = 0;
	}
	for (int i = 0; i < cNum; i++) {
		classTotalPosition[i][0] += samples[indexTracer->index][0];
		classTotalPosition[i][1] += samples[indexTracer->index][1];
		minimaxCenterIndex[i] = indexTracer->index;
		minimaxResultClass[indexTracer->index] = i + 1;
		indexTracer = indexTracer->next;
	}

	// Classifying samples based on distances and sum up cluster positions for calculating cluster centers
	for (int i = 0; i < CENTER_NUM * SAMPLE_SIZE; i++) {
		if (minimaxResultClass[i] == 0) {
			int minIndex = 0;
			double min = distance(samples[i], samples[minimaxCenterIndex[0]]);
			for (int j = 1; j < cNum; j++) {
				double d = distance(samples[i], samples[minimaxCenterIndex[j]]);
				if (d < min) {
					min = d;
					minIndex = j;
				}
			}
			minimaxResultClass[i] = minIndex + 1;
			classTotalPosition[minIndex][0] += samples[i][0];
			classTotalPosition[minIndex][1] += samples[i][1];
		}
	}
	// Assign cluster centers
	minimaxClassNum = cNum;
	if (minimaxClassCenters) delete[] minimaxClassCenters;
	minimaxClassCenters = new double*[cNum];
	for (int i = 0; i < cNum; i++) {
		minimaxClassCenters[i] = new double[2];
		minimaxClassCenters[i][0] = classTotalPosition[i][0] / (CENTER_NUM * SAMPLE_SIZE / cNum);
		minimaxClassCenters[i][1] = classTotalPosition[i][1] / (CENTER_NUM * SAMPLE_SIZE / cNum);
	}
	// Delete pointers
	indexTracer = indexHead;
	while (indexTracer) {
		indexHead = indexTracer;
		indexTracer = indexTracer->next;
		delete indexHead;
	}
	indexTracer = NULL;
	indexHead = NULL;
	for (int i = 0; i < cNum; i++)
		delete[] classTotalPosition[i];
	delete[] classTotalPosition;
}

void kmeans(int cNum) {
	// Initialize variables and randomly select initial class centers
	int iteration = 0;
	bool done = false;
	int* cCounter = new int[cNum];
	double** cCenters = new double*[cNum];
	double** cPreCenters = new double*[cNum];
	double** cCenterTotal = new double*[cNum];
	printf("initial center index: \n");
	for (int i = 0; i < cNum; i++) {
		bool repeat = true;
		while (repeat) {
			repeat = false;
			int r = rand() % 30;
			for (int j = 0; j < i; j++) {
				if (cCounter[j] == r) {
					repeat = true;
					break;
				}
			}
			if(!repeat) cCounter[i] = r;
		}
		cCenters[i] = new double[2];
		cPreCenters[i] = new double[2];
		cCenterTotal[i] = new double[2];
		cCenters[i][0] = samples[cCounter[i]][0];
		cCenters[i][1] = samples[cCounter[i]][1];
		cPreCenters[i][0] = samples[cCounter[i]][0];
		cPreCenters[i][1] = samples[cCounter[i]][1];
		cCenterTotal[i][0] = 0;
		cCenterTotal [i][1] = 0;
		printf("%d  ", cCounter[i]);
	}
	printf("\n\n");
	for (int i = 0; i < cNum; i++) cCounter[i] = 0;
	if (kmeansClassCenters) {
		for (int i = 0; i < kmeansClassNum; i++)
			delete[] kmeansClassCenters[i];
		delete[] kmeansClassCenters;
		kmeansClassCenters = new double*[cNum];
	}
	else kmeansClassCenters = new double*[cNum];
	kmeansClassNum = cNum;

	// Each kmeans iteration
	while (!done && iteration < MAX_ITERATION) {
		// Initialize variables
		for (int i = 0; i < cNum; i++) {
			cCounter[i] = 0;
			cCenterTotal[i][0] = 0;
			cCenterTotal[i][1] = 0;
		}
		
		// Classify samples and sum up stats for eahc class
		for (int i = 0; i < CENTER_NUM * SAMPLE_SIZE; i++) {
			int index = 0;
			double min = distance(samples[i], cCenters[0]);
			for (int j = 1; j < cNum; j++) {
				double d = distance(samples[i], cCenters[j]);
				if (d < min) {
					min = d;
					index = j;
				}
			}
			cCounter[index]++;
			cCenterTotal[index][0] += samples[i][0];
			cCenterTotal[index][1] += samples[i][1];
		}

		// Calculate class centers and check with previous class centers
		done = true;
		for (int i = 0; i < cNum; i++) {
			cCenters[i][0] = cCenterTotal[i][0] / cCounter[i];
			cCenters[i][1] = cCenterTotal[i][1] / cCounter[i];
			if (done && (cCenters[i][0] != cPreCenters[i][0] || cCenters[i][1] != cPreCenters[i][1]))
				done = false;
		}

		// Update previous class centers if the iteration is not done
		if (!done) {
			for (int i = 0; i < cNum; i++) {
				cPreCenters[i][0] = cCenters[i][0];
				cPreCenters[i][1] = cCenters[i][1];
			}
		}
		iteration++;
	}

	// Assign kmeans stats result
	kmeansIterationNum = iteration;
	for (int i = 0; i < cNum; i++) {
		kmeansClassCenters[i] = new double[2];
		kmeansClassCenters[i][0] = cCenters[i][0];
		kmeansClassCenters[i][1] = cCenters[i][1];
	}

	// Classify each sample since we did not store the result while running
	for (int i = 0; i < CENTER_NUM * SAMPLE_SIZE; i++) {
		int index = 0;
		double min = distance(samples[i], cCenters[0]);
		for (int j = 1; j < cNum; j++) {
			double d = distance(samples[i], cCenters[j]);
			if (d < min) {
				min = d;
				index = j;
			}
		}
		kmeansResultClass[i] = index + 1;
	}

	// Delete pointers
	delete[] cCounter;
	for (int i = 0; i < cNum; i++) {
		delete[] cCenters[i];
		delete[] cPreCenters[i];
		delete[] cCenterTotal[i];
	}
	delete[] cCenters;
	delete[] cPreCenters;
	delete[] cCenterTotal;
	cCounter = NULL;
	cCenters = NULL;
	cPreCenters = NULL;
	cCenterTotal = NULL;
}

double distance(double* a, double* b) {
	return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2));
}

centerIndexNode* appendToIndex(centerIndexNode* list, int index) {
	centerIndexNode* newNode = new centerIndexNode();
	newNode->index = index;
	newNode->next = NULL;
	centerIndexNode* h = list;
	if (!h) return newNode;
	while (h->next) h = h->next;
	h->next = newNode;
	return list;
}

void deletePointers() {
	if (minimaxClassCenters) {
		for (int i = 0; i < minimaxClassNum; i++) {
			delete[] minimaxClassCenters[i];
		}
		delete[] minimaxClassCenters;
	}
	if (kmeansClassCenters) {
		for (int i = 0; i < kmeansClassNum; i++) {
			delete[] kmeansClassCenters[i];
		}
		delete[] kmeansClassCenters;
	}
	if(minimaxCenterIndex)
		delete[] minimaxCenterIndex;
}

int userInput(int mode) {
	// mode 1: ask for radius for data gen
	// mode 2: ask for using minimax or kmeans
	// mode 3: ask for k for kmeans algo
	// mode 4: ask for exit
	string input;
	switch (mode){
	case 1:
		printf("For data generation, input the radius of the circle (1 - 15): ");
		break;
	case 2:
		printf("Input 1 or 2 to run minimax(1) or kmeans(2): ");
		break;
	case 3:
		printf("Input the number of k for running kmeans (2 - 7): ");
		break;
	case 4:
		printf("Done testing and want to exit the program? (y / n): ");
		break;
	}
	cin >> input;
	switch (mode) {
	case 1:
		while (input != "4" && input != "15" && input != "1" && input != "2" && input != "3" && input != "5" && input != "6" && input != "7" && input != "8" && input != "9" && input != "10" && input != "11" && input != "12" && input != "13" && input != "14") {
			printf("Incorrect input. For data generation, please input the number of the radius (1 - 15): ");
			cin >> input;
		}
		break;
	case 2:
		while (input != "1" && input != "2") {
			printf("Incorrect input. Please enter 1 to run minimax algo. or enter 2 to run kmeans algo (1 / 2): ");
			cin >> input;
		}
		break;
	case 3:
		while (input != "2" && input != "3" && input != "5" && input != "7" && input != "4" && input != "6") {
			printf("Incorrect input. Please enter number 2 to 7 to determine the number of classes in kmeas algo. (2 - 7): ");
			cin >> input;
		}
		break;
	case 4:
		while (input != "y" && input != "n" && input != "Y" && input != "N" && input != "yes" && input != "no" && input != "Yes" && input != "No") {
			printf("Incorrect input. Please enter 'y' to exit the program, or enter 'n' to keep testing (y / n): ");
			cin >> input;
		}
		if (input == "y" || input == "Y" || input == "yes" || input == "Yes") input = "1";
		else input = "2";	
		break;
	}
	return stoi(input);
}