//**************************************************************************
#ifndef GRAPH_H
#define GRAPH_H

//**************************************************************************
const int INF = 100000;

//**************************************************************************
class Graph //Adjacency List clas
{
private:
	int *A;
	int vertices;
public:
	Graph();
	~Graph();
	void fija_nverts(const int verts);
	void inserta_arista(const int ptA, const int ptB, const int edge);
	int arista(const int ptA, const int ptB);
	void imprime();
	void lee(char *filename);
	int get_vertices(){ return vertices; }
	inline int get_elemento_matriz_A(int i, int j){ return A[i*vertices + j]; }
};

//**************************************************************************
#endif
//**************************************************************************