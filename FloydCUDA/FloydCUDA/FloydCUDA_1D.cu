#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <time.h>
#include <algorithm>
#include "Graph.h"

#define blocksize 256

using namespace std;

//**************************************************************************
// Kernel to update the Matrix at k-th iteration
__global__ void floyd_kernel(int * M, const int nverts, const int k)
{

}



int main(int argc, char *argv[])
{
	// Control de errores de la entrada.
	if (argc != 2)
	{
		cerr << "Sintaxis: " << argv[0] << " <archivo de grafo>" << endl;
		return(-1);
	}


	//Get GPU information
	int devID;
	cudaDeviceProp props;
	cudaError_t err;

	// Obtenemos el ID del dispositivo que se va a usar y comprobamos si ha habido
	// alg�n tipo de error.
	err = cudaGetDevice(&devID);
	if (err != cudaSuccess) { cout << "Error while trying to get device ID." << endl; }

	// Obtenemos las propiedades del dispositivo correspondiente al ID obtenido anteriormente.
	cudaGetDeviceProperties(&props, devID);
	printf("Device %d: \"%s\" with Compute %d.%d capability\n",
		devID, props.name, props.major, props.minor);
	cout << "Numero de multiprocesadores: " << props.multiProcessorCount << endl;
	cout << "Numero max. de hebras por multiprocesador: " << props.maxThreadsPerMultiProcessor << endl;
	cout << "Numero hebras por Wrap: " << props.warpSize << endl;
	
	/****
	**
	** Inicializaci�n de los datos del algoritmo
	**
	****/

	// Leemos el fichero de entrada y lo guardamos en G.
	Graph G;
	G.lee(argv[1]);
	//cout << "EL Grafo de entrada es:"<<endl;
	//G.imprime();

	// Obtenemos el n�mero de vertices del problema.
	const int nverts = G.vertices;

	// Fijamos el n�mero de iteraciones del algoritmo.
	const int niters = nverts;

	// Calculamos el n�mero de elementos de la matriz.
	const int nverts2 = nverts*nverts;

	// Reservamos memoria en el HOST para la matriz de salida
	// en base al n�mero de elementos de la matriz.
	int *c_Out_M = new int[nverts2];

	// Calculamos el n�mero de bytes que ocupa la matriz de salida.
	int size = nverts2*sizeof(int);

	// Declaramos el punter� que apuntar� a la zona de memoria en DEVICE
	// que se emplear� para la matriz de entrada.
	int * d_In_M = NULL;



	/****
	**
	** Fase paralela del algoritmo. (Computaci�n en la GPU).
	**
	****/

	// Reservamos un n�mero de bytes de memoria DEVICE para la matriz
	// de entrada igual al n�mero de bytes que ocupa la matriz de salida
	// en memoria HOST.
	err = cudaMalloc((void **)&d_In_M, size);
	if (err != cudaSuccess) { cout << "ERROR: Bad Allocation in Device Memory" << endl; }

	// Tomamos la primera media del tiempo.
	double  t1 = clock();

	// Copiamos en la matriz alojada en memoria DEVICE la matriz de datos
	// le�da del fichero de entrada en memoria del HOST.
	err = cudaMemcpy(d_In_M, G.Get_Matrix(), size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { cout << "ERROR: COPY MATRIX TO DEVICE" << endl; }

	// Bucle principal del algoritmo.
	for (int k = 0; k<niters; k++)
	{

		//*******************************************************************

		//Kernel Launch 



		//*******************************************************************  

		// Comprobamos si se ha producido alg�n error en la iteraci�n k-�sima.
		err = cudaGetLastError();

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch kernel!\n");
			exit(EXIT_FAILURE);
		}

	}

	// Cuando finaliza el bucle principal del algoritmo, copiamos en la
	// matriz de salida en memoria HOST, la matriz que se ubica en memoria
	// DEVICE, que contiene la soluci�n.
	cudaMemcpy(c_Out_M, d_In_M, size, cudaMemcpyDeviceToHost);

	// Tomamos la segunda medida de tiempo.
	double Tgpu = clock();

	// Liberamos la memoria reservada previamente en DEVICE.
	err = cudaFree(d_In_M);
	if (err != cudaSuccess) { cout << "ERROR: Bad Release in Device Memory" << endl; }

	// Calculamos el tiempo empleado en resolver el problema.
	Tgpu = (Tgpu - t1) / CLOCKS_PER_SEC;
	cout << "Tiempo gastado GPU= " << Tgpu << endl << endl;



	/****
	**
	**	Versi�n secuencial.
	**
	****/

	// Tomamos la primera medida de tiempo.
	t1 = clock();

	// Bucle principal del algoritmo.
	for (int k = 0; k<niters; k++)
		for (int i = 0; i<nverts; i++)
			for (int j = 0; j<nverts; j++)
				if (i != j && i != k && j != k)
				{
					int vikj = min(G.arista(i, k) + G.arista(k, j), G.arista(i, j));
					G.inserta_arista(i, j, vikj);
				}

	// Tomamos la segunda medida de tiempo.
	double t2 = clock();

	// Calculamos el tiempo empleado en resolver el problema
	// secuencial.
	t2 = (t2 - t1) / CLOCKS_PER_SEC;

	//  cout << endl<<"EL Grafo con las distancias de los caminos m�s cortos es:"<<endl<<endl;
	//  G.imprime();
	cout << "Tiempo gastado CPU= " << t2 << endl << endl;
	cout << "Ganancia= " << t2 / Tgpu << endl;

	// Por �ltimo comprobamos que los resultados de la versi�n de CUDA y
	// la versi�n secuencial es la misma.
	for (int i = 0; i<nverts; i++)
		for (int j = 0; j<nverts; j++)
			if (abs(c_Out_M[i*nverts + j] - G.arista(i, j))>0)
				cout << "Error (" << i << "," << j << ")   "
				<< c_Out_M[i*nverts + j] << "..." << G.arista(i, j) << endl;

	// Liberamos la memoria empleada por la matriz de salida.
	delete[] c_Out_M;
}



