#ifndef VOXL_CONFIGURE_TFLITE_H
#define VOXL_CONFIGURE_TFLITE_H

static int _parse_opts(int argc, char* const argv[], int help_only);

static void _print_usage(void);

// Function to convert file to cJSON object
int parse_config(void);

#endif // VOXL_CONFIGURE_TFLITE_H