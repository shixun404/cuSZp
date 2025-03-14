#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cuSZp_utility.h>
#include <cuSZp_entry_f32.h>
#include <cuSZp_timer.h>


// void SZx_writeByteData(unsigned char *bytes, size_t byteLength, char *tgtFilePath)
// {
// 	FILE *pFile = fopen(tgtFilePath, "wb");
//     if (pFile == NULL)
//     {
//         printf("Failed to open input file. 3\n");
//         return;
//     }
    
//     fwrite(bytes, 1, byteLength, pFile); //write outSize bytes
//     fclose(pFile);
//     printf("Finished writing compressed data.\n");
// }

// void SZx_writeFloatData_inBytes(float *data, size_t nbEle, char* tgtFilePath)
// {
// 	size_t i = 0; 
// 	int state = SZx_SCES;
// 	lfloat buf;
// 	unsigned char* bytes = (unsigned char*)malloc(nbEle*sizeof(float));
// 	for(i=0;i<nbEle;i++)
// 	{
// 		buf.value = data[i];
// 		bytes[i*4+0] = buf.byte[0];
// 		bytes[i*4+1] = buf.byte[1];
// 		bytes[i*4+2] = buf.byte[2];
// 		bytes[i*4+3] = buf.byte[3];					
// 	}

// 	size_t byteLength = nbEle*sizeof(float);
// 	SZx_writeByteData(bytes, byteLength, tgtFilePath, &state);
// 	free(bytes);
// 	printf("Finished writing decompressed data.\n");
// }

template <typename T>
void write_array_to_binary(const std::string& fname, T* const _a, size_t const dtype_dataTypeLen)
{
    std::ofstream ofs(fname.c_str(), std::ios::binary | std::ios::out);
    if (not ofs.is_open()) return;
    ofs.write(reinterpret_cast<const char*>(_a), std::streamsize(dtype_dataTypeLen * sizeof(T)));
    ofs.close();
}

template <typename T>
T* read_binary_to_new_array(const std::string& fname, size_t dtype_dataTypeLen)
{
    std::ifstream ifs(fname.c_str(), std::ios::binary | std::ios::in);
    if (not ifs.is_open()) {
        std::cerr << "fail to open " << fname << std::endl;
        exit(1);
    }
    auto _a = new T[dtype_dataTypeLen]();
    ifs.read(reinterpret_cast<char*>(_a), std::streamsize(dtype_dataTypeLen * sizeof(T)));
    ifs.close();
    return _a;
}

int main(int argc, char* argv[])
{
    // Read input information.
    char oriFilePath[640];
    char errorMode[20];
    int status=0;
    if(argc != 4)
    {
        printf("Usage: cuSZp_gpu_f32_api [srcFilePath] [errorMode] [errBound] # errorMode can only be ABS or REL\n");
        printf("Example: cuSZp_gpu_f32_api testfloat_8_8_128.dat ABS 1E-2     # compress dataset with absolute 1E-2 error bound\n");
        printf("         cuSZp_gpu_f32_api testfloat_8_8_128.dat REL 1e-3     # compress dataset with relative 1E-3 error bound\n");
        exit(0);
    }
    
    sprintf(oriFilePath, "%s", argv[1]);
    sprintf(errorMode, "%s", argv[2]);
    std::string compFilePath(oriFilePath), decompFilePath(oriFilePath);
    compFilePath.append(".cuszpa");
    decompFilePath.append(".cuszpx");
    float errorBound = atof(argv[3]);

    // For measuring the end-to-end throughput.
    TimingGPU timer_GPU;

    // Input data preparation on CPU.
    float* oriData = NULL;
    float* decData = NULL;
    unsigned char* cmpBytes = NULL;
    size_t nbEle = 0;
    size_t cmpSize = 0;
    oriData = readFloatData_Yafan(oriFilePath, &nbEle, &status);
    decData = (float*)malloc(nbEle*sizeof(float));
    cmpBytes = (unsigned char*)malloc(nbEle*sizeof(float));

    // Generating error bounds.
    if(strcmp(errorMode, "REL")==0)
    {
        float max_val = oriData[0];
        float min_val = oriData[0];
        for(size_t i=0; i<nbEle; i++)
        {
            if(oriData[i]>max_val)
                max_val = oriData[i];
            else if(oriData[i]<min_val)
                min_val = oriData[i];
        }
        errorBound = errorBound * (max_val - min_val);
    }
    else if(strcmp(errorMode, "ABS")!=0)
    {
        printf("invalid errorMode! errorMode can only be ABS or REL.\n");
        exit(0);
    }
    
    // Input data preparation on GPU.
    float* d_oriData;
    float* d_decData;
    unsigned char* d_cmpBytes;
    // size_t pad_nbEle = (nbEle + 262144 - 1) / 262144 * 262144; // A temp demo, will add more block sizes in future implementation.
    size_t pad_nbEle = nbEle; // A temp demo, will add more block sizes in future implementation.
    
    cudaMalloc((void**)&d_oriData, sizeof(float)*pad_nbEle);
    cudaMemcpy(d_oriData, oriData, sizeof(float)*pad_nbEle, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_decData, sizeof(float)*pad_nbEle);
    // cudaMemset(d_decData, 0, sizeof(float)*pad_nbEle);
    cudaMalloc((void**)&d_cmpBytes, sizeof(float)*pad_nbEle);

    // Initializing CUDA Stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Just a warmup.
    for(int i=0; i<3; i++)
        SZp_compress_deviceptr_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize, errorBound, stream);
    
    // cuSZp compression.
    timer_GPU.StartCounter(); // set timer
    SZp_compress_deviceptr_f32(d_oriData, d_cmpBytes, nbEle, &cmpSize, errorBound, stream);
    float cmpTime = timer_GPU.GetCounter();
    
    // cudaMemcpy(cmpBytes, d_cmpBytes, cmpSize*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // write_array_to_binary(compFilePath, cmpBytes, cmpSize);
    // free(cmpBytes);
    // printf("asdassd\n");
    // cmpBytes = read_binary_to_new_array<unsigned char>(compFilePath, cmpSize);
    // cudaMemcpy(d_cmpBytes, cmpBytes, cmpSize*sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    // // cuSZp decompression.
    timer_GPU.StartCounter(); // set timer
    SZp_decompress_deviceptr_f32(d_decData, d_cmpBytes, nbEle, cmpSize, errorBound, stream);
    float decTime = timer_GPU.GetCounter();

    
    // Print result.
    printf("cuSZp finished!\n");
    printf("cuSZp compression   end-to-end speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/cmpTime);
    printf("cuSZp decompression end-to-end speed: %f GB/s\n", (nbEle*sizeof(float)/1024.0/1024.0)/decTime);
    printf("cuSZp compression ratio: %f\n", (nbEle*sizeof(float)/1024.0/1024.0)/(cmpSize*sizeof(unsigned char)/1024.0/1024.0));
    
    
    // Error check
    cudaMemcpy(cmpBytes, d_cmpBytes, cmpSize*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(decData, d_decData, sizeof(float)*nbEle, cudaMemcpyDeviceToHost);
    
    // PSNR
    double *result;
    result = computePSNR(nbEle, oriData, decData);
    printf("cuSZp PSNR: %f\n\n", result[2]);
    
    int not_bound = 0;
    for(size_t i=0; i<nbEle; i+=1)
    {
        if(abs(oriData[i]-decData[i]) > errorBound*1.1)
        {
            not_bound++;
            // printf("not bound: %zu oriData: %f, decData: %f, errors: %f, bound: %f\n", i, oriData[i], decData[i], abs(oriData[i]-decData[i]), errBound);
        }
    }
    
    
    // SZx_writeByteData(cmpBytes, cmpSize, outputFilePath);
    
    // SZx_writeFloatData_inBytes(decData, nbEle, outputFilePath);
    write_array_to_binary(compFilePath, cmpBytes, cmpSize);
    write_array_to_binary(decompFilePath, decData, nbEle);
    if(!not_bound) printf("\033[0;32mPass error check!\033[0m\n");
    else printf("\033[0;31mFail error check!\033[0m\n");
    
    // Free allocated data.
    free(oriData);
    free(decData);
    free(cmpBytes);
    cudaFree(d_oriData);
    cudaFree(d_decData);
    cudaFree(d_cmpBytes);
    cudaStreamDestroy(stream);
    return 0;
}
