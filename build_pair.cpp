#include <iostream>
#include <map>
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>
#include <set>
using namespace std;

map<string, int> ma;
string BuildLine(string name, int id, bool isTrain){
	char buf[256] = {0};
	int pid;
	if (ma.count(name))pid = ma[name];
	else{
		if (!isTrain)return "";
		pid = ma.size();
		ma[name] = pid;
	}
	sprintf(buf, "%s/%s_%.4d.jpg %d", name.c_str(), name.c_str(), id, pid);
	string s = buf;
	return s;	
}

void WriteFile(const char* pairFile,const char* outFile, bool isTrain){
	ofstream fout(outFile);
	ifstream fin(pairFile);
	int n;
	fin >> n;
	for (int i = 0;i < n;++i){
		string name;
		int id1, id2;
		fin >> name >> id1 >> id2;
		fout << BuildLine(name, id1, isTrain);
		fout << BuildLine(name, id2, isTrain);
	}
	for (int i = 0;i < n;++i){
		string name_a, name_b;
		int id_a, id_b;
		fin >> name_a >> id_a >> name_b >> id_b;
		fout << BuildLine(name_a, id_a, isTrain);
		fout << BuildLine(name_b, id_b, isTrain);
	}
}
int main(){
	WriteFile("./lfw-deepfunneled/pairsDevTrain.txt", "./lfw-deepfunneled/train.txt", true);
	cout << ma.size() << endl;
	WriteFile("./lfw-deepfunneled/pairsDevTest.txt", "./lfw-deepfunneled/val.txt", false);
	cout << ma.size() << endl;
	return 0;
}
