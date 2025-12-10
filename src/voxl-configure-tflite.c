/*******************************************************************************
 * Copyright 2025 ModalAI Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * 4. The Software is used solely in conjunction with devices provided by
 *    ModalAI Inc.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> 
#include <getopt.h>
#include <string.h>
#include <modal_json.h>
#include "voxl-configure-tflite.h"

#define CONFIG_PATH "/etc/modalai/voxl-tflite-server.conf"

cJSON *root;

static void _print_usage()
{
	printf("\nvoxl-configure-tflite is a command line configuration helper for voxl-tflite-server\n");
	printf("\nCommand line arguments are as follows:\n\n");
	printf("-h, --help             : Print this help message\n");
	printf("-s, --skip-frames      : How many frames in between inference frames\n");
	printf("-m, --model-path       : Full path to .tflite model file\n");
	printf("-p, --input-pipe       : Name of input camera pipe to run inference on\n");
	printf("-d, --delegate         : Hardware to run inference on (e.g. gpu)\n");
	printf("-r, --require-labels   : Require labels (true/false)\n");
	printf("-l, --label-path       : Full path to labels.txt file\n");
	printf("-a, --allow-multiple   : Allow multiple runtime instances (true/false)\n");
	printf("-o, --output-prefix    : Prefix for output pipe\n");
}


static int _parse_opts(int argc, char* const argv[], int help_only)       
{
	static struct option LongOptions[] =
	{
		{"help",          no_argument,        0, 'h'},
		{"skip-frames",   required_argument,  0, 's'},
		{"model-path",    required_argument,  0, 'm'},
		{"input-pipe",    required_argument,  0, 'p'},
		{"delegate",      required_argument,  0, 'd'},
		{"require-labels",required_argument,  0, 'r'},
		{"label-path",    required_argument,  0, 'l'},
		{"allow-multiple",required_argument,  0, 'a'},
		{"output-prefix", required_argument,  0, 'o'},
	};

	int optionIndex= 0;
	int status = 0;
	int option;

	while ((status == 0) && (option = getopt_long (argc, argv, "hs:m:p:d:r:l:a:o:", &LongOptions[0], &optionIndex)) != -1)
	{
		if (help_only) {
            if (option == 'h')
                return 1;
            else
                continue;  // ignore everything else
		}
		else {
			switch(option) {
				case 's':
					cJSON_ReplaceItemInObject(root, "skip_n_frames", cJSON_CreateNumber(atoi(optarg)));
					break;

				case 'm':
					cJSON_ReplaceItemInObject(root, "model", cJSON_CreateString(optarg));
					break;

				case 'p':
					cJSON_ReplaceItemInObject(root, "input_pipe", cJSON_CreateString(optarg));
					break;

				case 'd':
					cJSON_ReplaceItemInObject(root, "delegate", cJSON_CreateString(optarg));
					break;
				
				case 'r':
					if (strcmp(optarg, "true") == 0) {
						cJSON_ReplaceItemInObject(root, "requires_labels", cJSON_CreateBool(1));
					}
					else if (strcmp(optarg, "false") == 0) {
						cJSON_ReplaceItemInObject(root, "requires_labels", cJSON_CreateBool(0));
					}
					else {
						status = -1;
					}
					break;

				case 'l':
					cJSON_ReplaceItemInObject(root, "labels", cJSON_CreateString(optarg));
					break;
				
				case 'a':
					if (strcmp(optarg, "true") == 0) {
						cJSON_ReplaceItemInObject(root, "allow_multiple", cJSON_CreateBool(1));
					}
					else if (strcmp(optarg, "false") == 0) {
						cJSON_ReplaceItemInObject(root, "allow_multiple", cJSON_CreateBool(0));
					}
					else {
						status = -1;
					}
					break;				
				
				case 'o':
					cJSON_ReplaceItemInObject(root, "output_pipe_prefix", cJSON_CreateString(optarg));
					break;
					
				case '?':
				default:
					return 1;
			}
		}
		
	}

	return status;
}

int parse_config() {
  	FILE *fp = fopen(CONFIG_PATH, "r");
    if (fp == NULL) {
        return 1;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *buffer = (char *) malloc(file_size + 1);
    if (buffer == NULL) {
        perror("Memory allocation failed");
        fclose(fp);
        exit(1);
    }

	if (fread(buffer, 1, file_size, fp) != (unsigned long)file_size) {
    	perror("fread failed");
		exit(1);
	}

    buffer[file_size] = '\0';
    fclose(fp);

    char *start = buffer;
    while (*start && (*start == ' ' || *start == '\t' || *start == '\n' || *start == '\r'))
        start++;  // Skip whitespace

    if (start[0] == '/' && start[1] == '*') {
        start += 2; // Skip '/*'
        char *end_comment = strstr(start, "*/");
        if (end_comment != NULL) {
            start = end_comment + 2; // Move past '*/'
        }
    }

    // Skip any whitespace after the comment
    while (*start && (*start == ' ' || *start == '\t' || *start == '\n' || *start == '\r'))
        start++;

    root = cJSON_Parse(start);
    if (root == NULL) {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL) {
            fprintf(stderr, "Error before: %s\n", error_ptr);
        }
        free(buffer);
        exit(1);
    }

    free(buffer);
	return 0;
}

int main(int argc, char* const argv[])
{
	if (argc == 1) {
		int wizard_status = system("/usr/bin/voxl-configure-tflite-wizard");
		return wizard_status;
    }

	// Parse arguments to determine if user explicitly requested help
	if(_parse_opts(argc, argv, 1)){
		_print_usage();
		return 0;
	}

	// Parse config. If this fails, create a new cJSON root
	if (parse_config()) {
		root = cJSON_CreateObject();
		cJSON_AddNumberToObject(root, "skip_n_frames", 0);
		cJSON_AddStringToObject(root, "model", "/usr/bin/dnn/ssdlite_mobilenet_v2_coco.tflite");
		cJSON_AddStringToObject(root, "input_pipe", "run/mpa/hires_small_color");
		cJSON_AddStringToObject(root, "delegate", "gpu");
		cJSON_AddBoolToObject(root, "requires_labels", 1);
		cJSON_AddStringToObject(root, "labels", "usr/bin/dnn/coco_labels.txt");
		cJSON_AddBoolToObject(root, "allow_multiple", 1);
		cJSON_AddStringToObject(root, "output_pipe_prefix", "mobilenet");
	}

	// Parse all other arguments
	if(_parse_opts(argc, argv, 0)){
		_print_usage();
		return 1;
	}

	char *json_string = cJSON_Print(root);
    if (json_string == NULL) {
        fprintf(stderr, "Error converting cJSON to string.\n");
        cJSON_Delete(root);
        return 1;
    }

	FILE *fp = fopen(CONFIG_PATH, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file for writing.\n");
        free(json_string);
        cJSON_Delete(root);
        return 1;
    }

    fputs(json_string, fp);

    fclose(fp);
    free(json_string);
    cJSON_Delete(root);

    printf("JSON data successfully written to %s\n", CONFIG_PATH);

	return 0;
}



