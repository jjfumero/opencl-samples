
#include <stdio.h>
#include <stdlib.h>

#include "readSource.h"

char *readsource(const char *sourceFilename) {

    FILE *fp;
    int err;
    int size;
    char *source;
	
    fp = fopen(sourceFilename, "rb");
 
    if(fp == NULL) {
       printf("Could not open kernel file: %s\n", sourceFilename);
       exit(-1);
    }
 
    err = fseek(fp, 0, SEEK_END);
 
    if(err != 0) {
       printf("Error seeking to end of file\n");
       exit(-1);
 
    }
    size = ftell(fp);
 
    if(size < 0) {
       printf("Error getting file position\n");
       exit(-1);
    }
 
    err = fseek(fp, 0, SEEK_SET);
    if(err != 0) {
       printf("Error seeking to start of file\n");
       exit(-1);
 
    }
 
    source = (char*)malloc(size+1);
 
    if(source == NULL) {
       printf("Error allocating %d bytes for the program source\n", size+1);
       exit(-1);
    }
 
    err = fread(source, 1, size, fp);
    if(err != size) {
       printf("only read %d bytes\n", err);
       exit(0);
    }
 
    source[size] = '\0';
    return source;
 }


