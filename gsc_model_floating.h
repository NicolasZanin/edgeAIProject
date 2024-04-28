#define SINGLE_FILE
/**
  ******************************************************************************
  * @file    defines.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, Université Côte d'Azur, LEAT, France
  * @version 2.1.0
  * @date    10 january 2024
  * @brief   Global C pre-processor definitions to use to build all source files (incl. CMSIS-NN)
  */

/* CMSIS-NN round mode definition */
#if defined(WITH_CMSIS_NN) || defined(WITH_NMSIS_NN)



#error "Unrecognized round mode, only floor and nearest are supported by CMSIS-NN"

#endif // defined(WITH_CMSIS_NN) || defined(WITH_NMSIS_NN)
/**
  ******************************************************************************
  * @file    number.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    2 february 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __NUMBER_H__
#define __NUMBER_H__

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef TRAPV_SHIFT
#include <limits.h>
#include <stdio.h>
#include <assert.h>
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define _clamp_to(type, number) clamp_to_number_t_ ## type (number)
#define clamp_to(type, number) _clamp_to(type, number)
#define _scale(type, number, scale_factor, round_mode) scale_number_t_ ## type (number, scale_factor, round_mode)
#define scale(type, number, scale_factor, round_mode) _scale(type, number, scale_factor, round_mode)
#define _scale_and_clamp_to(type, number, scale_factor, round_mode) scale_and_clamp_to_number_t_ ## type (number, scale_factor, round_mode)
#define scale_and_clamp_to(type, number, scale_factor, round_mode) _scale_and_clamp_to(type, number, scale_factor, round_mode)

typedef enum {
  ROUND_MODE_NONE,
  ROUND_MODE_FLOOR,
  ROUND_MODE_NEAREST,
} round_mode_t;

// Idea 1: Write the smallest min max interval of the net, could be an issue for hybrid int type network
// Idea 2: listing any interval and add type in name in a switch case like <- better but painfull
// #define NUMBER_MIN		// Max value for this numeric type
// #define NUMBER_MAX		// Min value for this numeric type

// // Idea 1: List of all types and write any corresponding function 
// typedef  number_t;		// Standard size numeric type used for weights and activations
// typedef  long_number_t;	// Long numeric type used for intermediate results

#define NUMBER_MIN_INT32_T -2147483648
#define NUMBER_MAX_INT32_T 2147483647

static inline int64_t min_int32_t(
    int64_t a,
    int64_t b) {
	if (a <= b)
		return a;
	return b;
}

static inline int64_t max_int32_t(
    int64_t a,
    int64_t b) {
	if (a >= b)
		return a;
	return b;
}

static inline int64_t scale_number_t_int32_t(
  int64_t number, int scale_factor, round_mode_t round_mode) {


  if (scale_factor <= 0) {
#ifdef TRAPV_SHIFT
    // Check for possible overflow of left shift
    if (number > INT64_MAX >> -scale_factor) {
      fprintf(stderr,
              "Error: scale() overflow, number=%ld, scale_factor=%d, limit=%d\n",
              number,
              scale_factor,
              INT16_MAX >> -scale_factor);
      assert(number <= INT64_MAX >> -scale_factor);
    }
#endif
    // No rounding to apply when shifting left
    return number << - scale_factor;
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return number >> scale_factor;
  }
}
static inline int32_t clamp_to_number_t_int32_t(
  int64_t number) {
	return (int32_t) max_int32_t(
      NUMBER_MIN_INT32_T,
      min_int32_t(
        NUMBER_MAX_INT32_T, number));
}
static inline int32_t scale_and_clamp_to_number_t_int32_t(
  int64_t number, int scale_factor, round_mode_t round_mode) {
#ifdef WITH_CMSIS_NN
  // Not really CMSIS-NN but use SSAT anyway
  if (scale_factor <= 0) {
    // No rounding to apply when shifting left
    return __SSAT(number << - scale_factor, sizeof(int32_t) * 8);
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return __SSAT(number >> scale_factor, sizeof(int32_t) * 8);
  }
#else
  number = scale_number_t_int32_t(number, scale_factor, round_mode);
  return clamp_to_number_t_int32_t(number);
#endif
}

#define NUMBER_MIN_FLOAT -2147483648
#define NUMBER_MAX_FLOAT 2147483647

static inline float min_float(
    float a,
    float b) {
	if (a <= b)
		return a;
	return b;
}

static inline float max_float(
    float a,
    float b) {
	if (a >= b)
		return a;
	return b;
}

static inline float scale_number_t_float(
  float number, int scale_factor, round_mode_t round_mode) {
	return number;
}
static inline float clamp_to_number_t_float(
  float number) {
	return (float) number;
}
static inline float scale_and_clamp_to_number_t_float(
  float number, int scale_factor, round_mode_t round_mode) {
	return (float) number;
}





static inline void int64_t_to_float(int64_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int32_t_to_float(int32_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int16_t_to_float(int16_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}

static inline void int8_t_to_float(int8_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}
#endif //__NUMBER_H__

#ifdef __cplusplus
} // extern "C"
#endif
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_H_
#define _MAX_POOLING1D_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  1
#define INPUT_SAMPLES   16000
#define POOL_SIZE       12
#define POOL_STRIDE     12
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef float max_pooling1d_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  1
#define INPUT_SAMPLES   16000
#define POOL_SIZE       12
#define POOL_STRIDE     12
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void max_pooling1d(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#else
#error "Unsupported activation function"
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_H_
#define _CONV1D_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       1333
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef float conv1d_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       1333
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 0
#define BIASES_SCALE_FACTOR 0
#define TMP_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void conv1d(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

#error "Data type unsupported by CMSIS-NN"

#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    1
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const float  conv1d_bias[CONV_FILTERS] = {-0x1.c9c69a0000000p-8, 0x1.727e700000000p-6, 0x1.15b9ea0000000p-5, -0x1.ca240c0000000p-8, 0x1.8a31980000000p-5, 0x1.6e0d120000000p-5, 0x1.70519a0000000p-5, 0x1.9bce1c0000000p-5, 0x1.61c7c80000000p-5, 0x1.728fd40000000p-5, -0x1.26b49e0000000p-5, 0x1.c95ed60000000p-4, 0x1.3057380000000p-8, 0x1.bace460000000p-5, -0x1.0e91840000000p-8, -0x1.8d82fe0000000p-8}
;

const float  conv1d_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{0x1.84398e0000000p-1}
, {0x1.f5b8480000000p-3}
, {-0x1.bc05820000000p-1}
}
, {{0x1.91f7f40000000p-1}
, {-0x1.7233e80000000p-1}
, {-0x1.9cd0b20000000p+0}
}
, {{-0x1.1734da0000000p-1}
, {-0x1.651fd60000000p-5}
, {-0x1.cdfcfe0000000p-1}
}
, {{0x1.382bc20000000p-1}
, {0x1.397ba20000000p-1}
, {-0x1.346ffc0000000p-4}
}
, {{-0x1.8fdaf20000000p-1}
, {-0x1.b13f380000000p-3}
, {-0x1.211dcc0000000p+0}
}
, {{0x1.8eda6c0000000p-5}
, {-0x1.36c3b80000000p+0}
, {-0x1.613cbc0000000p+0}
}
, {{0x1.2187da0000000p-3}
, {-0x1.57b3f20000000p+0}
, {-0x1.5e3d300000000p+0}
}
, {{-0x1.f074de0000000p-2}
, {-0x1.2228400000000p+0}
, {-0x1.409f260000000p+0}
}
, {{-0x1.aba31e0000000p-1}
, {-0x1.fb43160000000p-1}
, {-0x1.f99b980000000p-1}
}
, {{-0x1.15e9da0000000p-2}
, {-0x1.f7f9760000000p-1}
, {-0x1.19a47a0000000p+0}
}
, {{0x1.4b86140000000p-2}
, {0x1.6ea7240000000p-1}
, {0x1.2c74740000000p-1}
}
, {{0x1.4be6380000000p-2}
, {0x1.f4746c0000000p-2}
, {0x1.3253ec0000000p-1}
}
, {{0x1.35acb20000000p-2}
, {0x1.2799680000000p-1}
, {0x1.38a3340000000p-2}
}
, {{-0x1.042a0a0000000p+0}
, {-0x1.a83c840000000p-1}
, {-0x1.ea09740000000p-1}
}
, {{-0x1.59c1420000000p-2}
, {-0x1.12781a0000000p-1}
, {0x1.ac56f00000000p-1}
}
, {{-0x1.b4fd020000000p-1}
, {0x1.7d658c0000000p-7}
, {0x1.9b1f800000000p-1}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_1_H_
#define _MAX_POOLING1D_1_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  16
#define INPUT_SAMPLES   1331
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef float max_pooling1d_1_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_1(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_1_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_1.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  16
#define INPUT_SAMPLES   1331
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void max_pooling1d_1(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#else
#error "Unsupported activation function"
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_1_H_
#define _CONV1D_1_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       332
#define CONV_FILTERS        30
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef float conv1d_1_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_1(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_1_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_1.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       332
#define CONV_FILTERS        30
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 0
#define BIASES_SCALE_FACTOR 0
#define TMP_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void conv1d_1(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

#error "Data type unsupported by CMSIS-NN"

#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    16
#define CONV_FILTERS      30
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const float  conv1d_1_bias[CONV_FILTERS] = {-0x1.e2d3d00000000p-4, 0x1.25a7a20000000p-6, -0x1.1dc3e00000000p-6, 0x1.84f8f60000000p-3, -0x1.9a51d00000000p-7, -0x1.0f5c6a0000000p-5, 0x1.9474600000000p-3, -0x1.b0a9a60000000p-5, -0x1.85f0560000000p-5, 0x1.93b4be0000000p-4, 0x1.64d9180000000p-3, -0x1.359ed20000000p-3, -0x1.0c74640000000p-4, 0x1.4b2cb20000000p-10, 0x1.b600f40000000p-3, 0x1.7470dc0000000p-3, 0x1.99e23a0000000p-7, 0x1.505a820000000p-3, -0x1.23abf20000000p-7, 0x1.5fab800000000p-3, 0x1.8db5aa0000000p-5, 0x1.6263c00000000p-4, -0x1.819a2c0000000p-4, -0x1.fc89cc0000000p-4, 0x1.b187600000000p-6, -0x1.2717da0000000p-3, -0x1.5dbda60000000p-3, 0x1.26a8700000000p-4, -0x1.496c500000000p-3, 0x1.d38a780000000p-5}
;

const float  conv1d_1_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{0x1.05570c0000000p+0, 0x1.6904060000000p-2, 0x1.0320220000000p-7, 0x1.07360a0000000p-1, -0x1.63c54c0000000p-6, 0x1.2d73a20000000p-4, 0x1.b387280000000p-2, 0x1.6b52ce0000000p-3, 0x1.13e2040000000p-2, 0x1.ffb10e0000000p-4, -0x1.2d0c6c0000000p-4, 0x1.1da71a0000000p-3, -0x1.0457220000000p-3, 0x1.d1c9380000000p-4, 0x1.4a19420000000p-1, 0x1.e035e60000000p-1}
, {0x1.030cac0000000p+0, 0x1.3a3e260000000p-2, -0x1.07e2dc0000000p-4, 0x1.9a3bc80000000p-2, 0x1.88c5c20000000p-6, 0x1.b57d520000000p-4, -0x1.3171000000000p-3, 0x1.7bec5a0000000p-3, -0x1.19d8aa0000000p-3, 0x1.4f07a20000000p-4, 0x1.98186e0000000p-3, -0x1.ae60020000000p-3, 0x1.c73ee80000000p-4, -0x1.05aebe0000000p-3, 0x1.79a6980000000p-4, 0x1.f9576a0000000p-1}
, {0x1.1b1eae0000000p-1, 0x1.4832940000000p-2, 0x1.5d753c0000000p-3, 0x1.f6f8060000000p-3, 0x1.a8793c0000000p-3, 0x1.6f57580000000p-3, 0x1.7bab320000000p-2, 0x1.4f46dc0000000p-2, 0x1.0fd66c0000000p-3, 0x1.65311c0000000p-3, -0x1.b0fe0c0000000p-5, -0x1.3008b80000000p-3, -0x1.b81faa0000000p-6, 0x1.fc8a520000000p-4, 0x1.c1469e0000000p-1, 0x1.00e4140000000p+0}
}
, {{0x1.64b2ca0000000p-2, 0x1.0a490e0000000p-2, 0x1.afe9080000000p-3, 0x1.d3989c0000000p-2, 0x1.0049280000000p-2, 0x1.58ce1e0000000p-3, 0x1.1b710c0000000p-5, 0x1.31134c0000000p-2, 0x1.8d71e00000000p-7, 0x1.a2e3de0000000p-2, -0x1.4f76e60000000p-2, -0x1.aa46f80000000p-5, 0x1.22832c0000000p-7, 0x1.e8c1880000000p-3, 0x1.9240000000000p-2, 0x1.0f36cc0000000p-2}
, {0x1.2f1d620000000p-3, 0x1.2917fe0000000p-3, 0x1.d8afa60000000p-2, -0x1.ef6a280000000p-3, 0x1.418cd00000000p-1, 0x1.0b63160000000p-3, 0x1.1951340000000p-1, 0x1.1b9cf20000000p-2, 0x1.44a8ac0000000p-2, 0x1.1a40f80000000p-2, -0x1.c946780000000p-2, -0x1.3cd6000000000p-4, -0x1.c5dac40000000p-5, 0x1.3076d00000000p-2, -0x1.c351b60000000p-5, 0x1.2e5fc40000000p-4}
, {0x1.16c15a0000000p-4, 0x1.c2c1800000000p-3, 0x1.869ed00000000p-2, -0x1.e3866a0000000p-3, 0x1.0fe8b20000000p-2, 0x1.29485a0000000p-2, 0x1.1fcc6e0000000p-1, 0x1.3d76ea0000000p-1, 0x1.7b36620000000p-2, 0x1.02d7b00000000p-1, -0x1.faf7780000000p-4, 0x1.357e720000000p-4, 0x1.1950f00000000p-7, 0x1.ed9e500000000p-2, 0x1.1d729a0000000p-1, 0x1.3eedf20000000p-1}
}
, {{0x1.eb8d380000000p-2, 0x1.9ba1c40000000p-2, 0x1.1c2ef60000000p-2, 0x1.36d4760000000p-3, 0x1.29d5800000000p-3, 0x1.34cae60000000p-2, 0x1.abc6a00000000p-2, 0x1.14a9460000000p-3, 0x1.48b0080000000p-5, 0x1.18ee380000000p-2, 0x1.9121980000000p-3, -0x1.70c0c20000000p-3, 0x1.14ca8c0000000p-5, 0x1.fbe3ec0000000p-5, -0x1.35c3680000000p-3, 0x1.12f8fc0000000p-2}
, {0x1.4689480000000p-2, 0x1.39fd980000000p-2, 0x1.5f51360000000p-3, -0x1.8956f80000000p-3, 0x1.b47f6c0000000p-3, 0x1.f20cb00000000p-2, 0x1.53b9fa0000000p-2, 0x1.00db2a0000000p-2, 0x1.1f24b80000000p-2, 0x1.b67b760000000p-2, -0x1.ab93d00000000p-3, 0x1.c9ba080000000p-7, -0x1.4f29660000000p-3, 0x1.53f23e0000000p-3, -0x1.6b6a440000000p-4, -0x1.b5e57c0000000p-5}
, {0x1.a3e6900000000p-3, 0x1.1a88320000000p-2, 0x1.b2c65a0000000p-2, -0x1.3ca8ba0000000p-3, 0x1.9bb9840000000p-3, 0x1.19ad160000000p-1, 0x1.38203e0000000p-2, 0x1.0a85c40000000p-1, 0x1.0509880000000p-1, 0x1.00a1e40000000p-1, 0x1.ca64f00000000p-3, -0x1.bd35bc0000000p-4, -0x1.f0a8ca0000000p-4, 0x1.31dc380000000p-2, 0x1.c56f920000000p-2, 0x1.18987e0000000p+0}
}
, {{-0x1.1955360000000p-4, -0x1.c063a60000000p-3, -0x1.76092c0000000p-5, 0x1.dc2b400000000p-3, -0x1.9839e60000000p-3, -0x1.3c0f480000000p-2, -0x1.0fcc3c0000000p-3, -0x1.e16f4e0000000p-3, -0x1.eae6a00000000p-3, -0x1.0fe22a0000000p-4, 0x1.019b360000000p-2, 0x1.b2e8d40000000p-2, 0x1.0bd20a0000000p-2, -0x1.52b5e20000000p-2, -0x1.4210f60000000p-4, -0x1.101f6a0000000p-2}
, {-0x1.197fe40000000p-3, -0x1.80a77e0000000p-2, -0x1.41f3420000000p-4, 0x1.2cea580000000p-5, -0x1.9ea3480000000p-4, -0x1.ca06920000000p-4, -0x1.dba1580000000p-3, -0x1.50bcb20000000p-4, -0x1.060b980000000p-2, -0x1.5b9b100000000p-2, 0x1.90a0e00000000p-3, 0x1.a4b3920000000p-2, 0x1.d589000000000p-3, -0x1.639d700000000p-2, -0x1.b8f6ce0000000p-3, -0x1.f6559a0000000p-2}
, {-0x1.5977f80000000p-3, -0x1.33df760000000p-5, 0x1.609da00000000p-3, 0x1.ca242c0000000p-3, -0x1.f39b680000000p-3, -0x1.4b6d960000000p-3, -0x1.9bf3420000000p-6, -0x1.5ca3660000000p-3, -0x1.857b080000000p-2, 0x1.9b3b9c0000000p-4, 0x1.281ea20000000p-1, 0x1.c7899a0000000p-2, 0x1.28f3920000000p-2, -0x1.31af960000000p-2, -0x1.a53a8a0000000p-5, -0x1.41074a0000000p-3}
}
, {{0x1.4b188a0000000p-1, 0x1.108d680000000p-6, -0x1.1dd5300000000p-3, 0x1.9d6b940000000p-3, -0x1.07d1b00000000p-6, -0x1.9bd9280000000p-3, -0x1.314cd60000000p-3, -0x1.2edaae0000000p-7, 0x1.0151480000000p-4, 0x1.482d1a0000000p-7, -0x1.1a94ce0000000p-5, -0x1.aa9cd00000000p-5, 0x1.93cc4a0000000p-3, -0x1.ea8d100000000p-3, 0x1.2e74940000000p-2, 0x1.8d23760000000p-1}
, {0x1.7a50bc0000000p-2, 0x1.210e7a0000000p-2, 0x1.a8e9460000000p-3, -0x1.1e4dc20000000p-5, 0x1.5434f20000000p-3, 0x1.093bfe0000000p-1, 0x1.ed41740000000p-2, 0x1.ad79d60000000p-2, 0x1.8b52fc0000000p-2, 0x1.5fca9e0000000p-2, -0x1.ed42220000000p-2, 0x1.024cd40000000p-6, 0x1.67517c0000000p-4, 0x1.13030a0000000p-2, -0x1.5526b40000000p-4, -0x1.48b21e0000000p-4}
, {0x1.9903440000000p-3, 0x1.c946880000000p-4, 0x1.82a5e60000000p-2, -0x1.1b778e0000000p-3, 0x1.07c12e0000000p-2, 0x1.c810ba0000000p-2, 0x1.2cc0100000000p-3, 0x1.aacd7a0000000p-2, 0x1.977f2a0000000p-2, 0x1.c171b00000000p-3, -0x1.7d6d680000000p-2, -0x1.4d56f20000000p-3, 0x1.0e39640000000p-3, 0x1.69616c0000000p-2, 0x1.3899a40000000p-1, 0x1.7c29e00000000p-1}
}
, {{0x1.9687140000000p-5, 0x1.5f65f60000000p-6, -0x1.887a240000000p-5, -0x1.58a36e0000000p-3, -0x1.f2fa600000000p-5, -0x1.8c3cfa0000000p-3, 0x1.6a3f660000000p-5, 0x1.8fc8880000000p-3, -0x1.90d16e0000000p-3, -0x1.64ae840000000p-3, -0x1.522d2a0000000p-3, -0x1.d6e71c0000000p-5, -0x1.82c0760000000p-3, -0x1.53f8f60000000p-3, 0x1.1484640000000p-7, -0x1.01c75c0000000p-3}
, {-0x1.b8b2720000000p-3, -0x1.48c97e0000000p-4, -0x1.6829c40000000p-3, -0x1.d0c0c00000000p-8, -0x1.51aab60000000p-4, -0x1.65f6960000000p-3, 0x1.f551a60000000p-4, -0x1.d59f760000000p-4, 0x1.52005a0000000p-5, 0x1.14d7d40000000p-3, -0x1.4cda280000000p-11, -0x1.0d4d720000000p-4, -0x1.c612000000000p-4, 0x1.e65b700000000p-7, -0x1.b349aa0000000p-4, -0x1.436adc0000000p-3}
, {-0x1.cc7bd80000000p-3, 0x1.4c4dde0000000p-4, -0x1.8e03d60000000p-4, -0x1.0b381c0000000p-4, -0x1.378fb00000000p-8, -0x1.876ca00000000p-3, -0x1.2408fa0000000p-4, 0x1.038e0e0000000p-5, 0x1.360d1e0000000p-4, -0x1.628c980000000p-3, -0x1.656bf60000000p-4, 0x1.26f7740000000p-3, -0x1.c555b20000000p-3, 0x1.5ad20c0000000p-5, -0x1.3fcdde0000000p-3, 0x1.3ab85a0000000p-5}
}
, {{0x1.14ad380000000p-2, -0x1.641d280000000p-2, -0x1.007f900000000p-4, 0x1.f725460000000p-3, -0x1.6cfe8a0000000p-2, -0x1.5c2b700000000p-2, -0x1.ee40300000000p-3, -0x1.3840c60000000p-1, -0x1.d716a60000000p-2, -0x1.4547b20000000p-3, -0x1.8f460e0000000p-4, 0x1.5cc6b40000000p-2, 0x1.a9c73c0000000p-5, -0x1.d422020000000p-2, -0x1.52775e0000000p-4, -0x1.05a9400000000p-2}
, {0x1.2c71920000000p-4, -0x1.a84fb60000000p-2, -0x1.30720c0000000p-4, -0x1.15d6000000000p-3, -0x1.5ec32c0000000p-1, -0x1.c489c80000000p-4, -0x1.5dcd400000000p-2, -0x1.74cc700000000p-2, -0x1.8721b60000000p-1, -0x1.db72120000000p-3, -0x1.5bb67a0000000p-4, 0x1.17ccd40000000p-4, 0x1.260bde0000000p-5, -0x1.d8660e0000000p-2, 0x1.1e57b80000000p-3, 0x1.5f336a0000000p-2}
, {-0x1.3c771c0000000p-2, -0x1.10dff60000000p-2, 0x1.86ff7e0000000p-6, -0x1.8d4f6e0000000p-6, -0x1.1f08e20000000p-1, -0x1.06bc1a0000000p-2, -0x1.a51d680000000p-3, -0x1.6333c00000000p-2, -0x1.6ae5f20000000p-2, -0x1.b6fbd40000000p-2, -0x1.728cc00000000p-4, 0x1.8a67f20000000p-2, 0x1.56088c0000000p-2, -0x1.ded61e0000000p-2, -0x1.7335f60000000p-3, 0x1.b4384e0000000p-2}
}
, {{0x1.443d840000000p-3, 0x1.af679e0000000p-2, 0x1.a7c7f80000000p-2, 0x1.01b13e0000000p-5, 0x1.1375220000000p-2, 0x1.23c91c0000000p-1, 0x1.57508e0000000p-1, 0x1.1f8a960000000p-2, 0x1.4efbac0000000p-2, 0x1.019f6e0000000p-1, -0x1.f4e12e0000000p-4, 0x1.41e4120000000p-3, 0x1.c689860000000p-3, 0x1.07440e0000000p-1, 0x1.8b63dc0000000p-2, 0x1.60c8c60000000p-2}
, {0x1.912ab20000000p-4, -0x1.23fd720000000p-1, -0x1.95c5d20000000p-2, 0x1.5430340000000p-2, -0x1.cc858c0000000p-2, -0x1.7753540000000p-3, -0x1.0938fe0000000p-2, -0x1.0bc4d60000000p-2, -0x1.5862c40000000p-4, -0x1.fe3b0c0000000p-4, 0x1.15a2a80000000p-4, 0x1.026af20000000p-2, -0x1.0136440000000p-4, -0x1.2a16520000000p-2, 0x1.c4d6b80000000p-2, 0x1.2efeb60000000p-1}
, {0x1.cd5ee00000000p-3, -0x1.3ef7320000000p-2, -0x1.4a93300000000p-1, 0x1.6c4ad40000000p-4, -0x1.0132560000000p-1, -0x1.6beb6c0000000p-1, -0x1.e973020000000p-2, -0x1.21ef4c0000000p-1, -0x1.7f77680000000p-2, -0x1.a9a0940000000p-1, 0x1.3dd98e0000000p-4, 0x1.82ca7c0000000p-3, 0x1.438c120000000p-2, -0x1.7afe220000000p-2, 0x1.e8d2520000000p-4, 0x1.4822560000000p-1}
}
, {{-0x1.dd3f4c0000000p-4, -0x1.521bae0000000p-3, 0x1.37886a0000000p-3, 0x1.797a4a0000000p-4, 0x1.1b2dee0000000p-3, 0x1.2ee72e0000000p-6, 0x1.5ceb7e0000000p-4, -0x1.8521ca0000000p-3, -0x1.3cf16a0000000p-4, -0x1.42b1ee0000000p-4, 0x1.0ec2bc0000000p-3, -0x1.8b39e00000000p-4, -0x1.e306f00000000p-4, 0x1.40e6460000000p-3, 0x1.26d7660000000p-6, 0x1.ac7d680000000p-5}
, {0x1.fe81600000000p-4, -0x1.19dd320000000p-4, 0x1.6b21740000000p-5, 0x1.0afb740000000p-4, -0x1.d2513c0000000p-4, -0x1.3145520000000p-3, -0x1.4b99240000000p-5, 0x1.0a5c060000000p-5, -0x1.9a6c240000000p-3, -0x1.5d57b60000000p-6, -0x1.1e85360000000p-4, -0x1.dd63980000000p-3, -0x1.72f8040000000p-3, -0x1.1182380000000p-3, 0x1.40cb400000000p-4, -0x1.ca5d280000000p-4}
, {0x1.8f2d1e0000000p-5, -0x1.5ea3780000000p-5, -0x1.2be5780000000p-4, -0x1.a872940000000p-3, -0x1.8657b20000000p-5, -0x1.57af4a0000000p-3, 0x1.2735e40000000p-4, -0x1.ef38000000000p-4, 0x1.aa521c0000000p-4, -0x1.79d95a0000000p-7, -0x1.1211480000000p-4, -0x1.47766c0000000p-4, 0x1.0fbf5a0000000p-3, -0x1.a8db600000000p-5, 0x1.6fd7240000000p-8, 0x1.770fae0000000p-5}
}
, {{0x1.f406a60000000p-2, -0x1.a06d6a0000000p-2, -0x1.f5944a0000000p-2, 0x1.b329280000000p-3, -0x1.35c7d40000000p-1, -0x1.2b3ca20000000p-1, -0x1.20f8f40000000p-1, -0x1.4d3d7c0000000p-1, -0x1.d5d9a00000000p-3, -0x1.140a040000000p-1, 0x1.46b6100000000p-4, 0x1.037cac0000000p-2, 0x1.97b2820000000p-2, -0x1.ddf5f80000000p-3, 0x1.32976a0000000p-5, 0x1.9ae9340000000p-3}
, {-0x1.c04c760000000p-5, 0x1.3bf5fc0000000p-3, 0x1.7ff0c80000000p-4, 0x1.c2f1240000000p-4, -0x1.0890960000000p-2, 0x1.2dfbdc0000000p-4, -0x1.3c95d60000000p-5, -0x1.98b8980000000p-3, -0x1.8fbb780000000p-3, 0x1.1f72920000000p-5, -0x1.0f7bbc0000000p-2, 0x1.2431b00000000p-2, 0x1.2b617e0000000p-3, -0x1.2ade620000000p-2, 0x1.37c3480000000p-2, 0x1.6b20cc0000000p-1}
, {0x1.21d5f20000000p-2, 0x1.374b260000000p-4, -0x1.5f4eae0000000p-3, 0x1.c4e3a20000000p-3, -0x1.112fae0000000p-2, 0x1.07de360000000p-3, 0x1.cbf78e0000000p-4, 0x1.750dda0000000p-5, -0x1.ada84c0000000p-5, -0x1.a439fa0000000p-6, 0x1.0a72980000000p-4, 0x1.8955140000000p-2, 0x1.d011ee0000000p-2, -0x1.1ad5d60000000p-5, -0x1.0905c20000000p-1, -0x1.24687e0000000p-3}
}
, {{-0x1.7200b20000000p-2, -0x1.53bfa80000000p-3, -0x1.f7814e0000000p-8, 0x1.fe28da0000000p-4, 0x1.3490160000000p-6, -0x1.d907cc0000000p-4, -0x1.267e560000000p-2, -0x1.3961ae0000000p-3, -0x1.95929a0000000p-6, -0x1.56e98e0000000p-3, 0x1.ca95da0000000p-2, 0x1.5d91360000000p-2, 0x1.5528f80000000p-2, -0x1.a898d80000000p-2, -0x1.980e880000000p-4, -0x1.bab42c0000000p-2}
, {-0x1.46b0180000000p-2, -0x1.8abca80000000p-2, -0x1.fcaede0000000p-5, 0x1.6f41120000000p-2, 0x1.3deed00000000p-5, -0x1.7e28280000000p-2, -0x1.052dc60000000p-2, -0x1.0a639a0000000p-2, -0x1.e503280000000p-4, -0x1.7807100000000p-4, 0x1.14d0f40000000p-2, 0x1.d845820000000p-3, 0x1.7b00be0000000p-2, -0x1.e28dfc0000000p-3, -0x1.f794900000000p-2, -0x1.29f39c0000000p-1}
, {-0x1.28fa460000000p-3, -0x1.cf71600000000p-3, 0x1.4486d00000000p-3, 0x1.a8a6b20000000p-4, -0x1.50fbb20000000p-4, -0x1.0e73d20000000p-3, -0x1.5d137e0000000p-2, -0x1.dd55760000000p-4, -0x1.154e3e0000000p-3, -0x1.5a287a0000000p-4, 0x1.0c81920000000p-1, 0x1.ec4d440000000p-4, 0x1.f525460000000p-3, -0x1.e397440000000p-3, -0x1.4bd6e60000000p-2, -0x1.cdba200000000p-2}
}
, {{0x1.305d5c0000000p+0, 0x1.18deb80000000p-2, -0x1.9470ca0000000p-3, 0x1.8aa4980000000p-2, 0x1.06c41e0000000p-3, 0x1.390a320000000p-4, 0x1.493ad80000000p-4, -0x1.fc13160000000p-4, 0x1.d842aa0000000p-11, 0x1.e17b400000000p-3, 0x1.3fc32a0000000p-4, 0x1.51cc340000000p-4, -0x1.ff40dc0000000p-4, -0x1.17f9260000000p-8, 0x1.9c3dd00000000p-2, 0x1.cae9640000000p-1}
, {0x1.c060da0000000p-3, 0x1.a49d8a0000000p-2, 0x1.6201020000000p-3, 0x1.5f41980000000p-3, 0x1.b1a72c0000000p-3, 0x1.30c3e00000000p-2, 0x1.58ab740000000p-3, 0x1.421b4c0000000p-2, 0x1.b92e6c0000000p-5, 0x1.57bd1a0000000p-3, -0x1.85cf980000000p-5, -0x1.28a26a0000000p-3, -0x1.6eef960000000p-3, 0x1.75b6fe0000000p-3, 0x1.abcee20000000p-2, 0x1.22079e0000000p+0}
, {0x1.ce9a800000000p-1, 0x1.0645dc0000000p-1, 0x1.82e22a0000000p-3, 0x1.623d580000000p-2, 0x1.80da040000000p-4, 0x1.8f18c00000000p-2, 0x1.b98bde0000000p-2, 0x1.ff2d820000000p-3, 0x1.a44f0c0000000p-3, 0x1.69bcde0000000p-3, -0x1.cf749e0000000p-4, -0x1.f910780000000p-3, 0x1.cae40e0000000p-3, 0x1.b2e8340000000p-4, 0x1.efca720000000p-2, 0x1.6d4f360000000p-1}
}
, {{-0x1.c491b40000000p-3, -0x1.26491c0000000p-7, 0x1.bb35f20000000p-4, 0x1.61626e0000000p-5, -0x1.9532a00000000p-6, -0x1.a545160000000p-3, 0x1.5214f00000000p-6, 0x1.ab63340000000p-4, 0x1.14ff300000000p-8, -0x1.6c5bac0000000p-3, -0x1.8dfec20000000p-3, 0x1.dc029a0000000p-6, -0x1.054fb40000000p-2, 0x1.8b27ac0000000p-4, 0x1.0fabda0000000p-9, -0x1.4a62220000000p-3}
, {-0x1.5713a20000000p-3, -0x1.9f2b220000000p-4, 0x1.a3bcf80000000p-4, -0x1.73e7b00000000p-5, -0x1.87c1740000000p-4, -0x1.0973aa0000000p-4, -0x1.1660e20000000p-3, 0x1.935d400000000p-8, 0x1.0969520000000p-4, -0x1.c7b0920000000p-3, -0x1.17e02e0000000p-2, -0x1.57c1920000000p-4, -0x1.5733320000000p-3, -0x1.09af940000000p-4, 0x1.cbb46e0000000p-5, 0x1.60c47a0000000p-4}
, {-0x1.5016e20000000p-3, 0x1.6f3aaa0000000p-4, 0x1.e79a6c0000000p-4, -0x1.88280c0000000p-3, -0x1.5aecf00000000p-3, -0x1.51b8700000000p-4, 0x1.209f440000000p-3, 0x1.1b25ea0000000p-3, 0x1.2e8b4c0000000p-3, -0x1.18cb5c0000000p-6, -0x1.8e2c2a0000000p-3, -0x1.8ff2520000000p-3, 0x1.1e5b500000000p-4, -0x1.7005e60000000p-4, -0x1.aaf3740000000p-4, 0x1.51ad020000000p-3}
}
, {{0x1.76d9e00000000p-3, 0x1.33dad40000000p-2, 0x1.61a4a40000000p-2, 0x1.cd2d400000000p-5, 0x1.0eec160000000p-3, 0x1.cd8a7e0000000p-2, 0x1.1471960000000p-3, 0x1.e8c3de0000000p-3, 0x1.9134f40000000p-2, 0x1.7a70160000000p-2, -0x1.6200f20000000p-1, -0x1.dbe3860000000p-4, -0x1.73b2b60000000p-6, 0x1.ae17a40000000p-2, -0x1.f62c680000000p-3, -0x1.84d9220000000p-8}
, {-0x1.2c69f00000000p-1, 0x1.3671c80000000p-2, 0x1.8192ec0000000p-2, -0x1.33921e0000000p-4, 0x1.f0b0560000000p-2, 0x1.e4c1e00000000p-2, 0x1.ad62980000000p-3, 0x1.92c7d40000000p-2, 0x1.bd8cf80000000p-2, 0x1.e9108c0000000p-2, -0x1.dc22a00000000p-2, -0x1.821eda0000000p-5, 0x1.0068d40000000p-5, 0x1.880da80000000p-3, 0x1.623dc60000000p-3, 0x1.6920120000000p-2}
, {0x1.61545a0000000p-3, 0x1.f5a20a0000000p-3, -0x1.7319020000000p-3, 0x1.5e677a0000000p-1, -0x1.f734780000000p-7, 0x1.0fa9220000000p-2, -0x1.1d7dc80000000p-4, -0x1.768fa60000000p-3, -0x1.175bb60000000p-4, -0x1.35196a0000000p-4, 0x1.355b840000000p-4, -0x1.24bc3c0000000p-7, 0x1.657ed00000000p-3, 0x1.9397d60000000p-8, 0x1.119fa40000000p-2, 0x1.7113120000000p-1}
}
, {{-0x1.147d8a0000000p-3, -0x1.dc85120000000p-3, 0x1.a6320e0000000p-4, 0x1.2bb51e0000000p-4, -0x1.15f5a60000000p-2, -0x1.98b1280000000p-3, -0x1.21fe0c0000000p-2, -0x1.4fe88e0000000p-3, 0x1.938a2c0000000p-5, 0x1.56d32c0000000p-4, 0x1.4488d00000000p-3, 0x1.14c7bc0000000p-3, 0x1.1f8ed60000000p-3, -0x1.3988040000000p-2, -0x1.570d9c0000000p-2, -0x1.ebf92e0000000p-2}
, {-0x1.b764760000000p-2, -0x1.877b140000000p-3, -0x1.db288c0000000p-7, 0x1.6aafea0000000p-10, -0x1.2901ae0000000p-2, -0x1.f71c580000000p-3, -0x1.7813d20000000p-2, -0x1.36dcb60000000p-4, -0x1.1cbeba0000000p-6, -0x1.e18fb20000000p-3, 0x1.efdd140000000p-6, 0x1.cb24c20000000p-3, -0x1.dfca380000000p-5, -0x1.31fab60000000p-2, -0x1.7b54d60000000p-2, -0x1.c8fd8a0000000p-2}
, {-0x1.8bb5420000000p-2, -0x1.3eafd40000000p-2, 0x1.30fd980000000p-3, 0x1.8b99dc0000000p-5, 0x1.c7d8fa0000000p-4, 0x1.2e066e0000000p-4, -0x1.a260120000000p-3, -0x1.0aeb140000000p-2, -0x1.d804120000000p-3, -0x1.765d060000000p-3, 0x1.57ff800000000p-2, 0x1.ab47260000000p-2, -0x1.afc1a60000000p-5, -0x1.8c07720000000p-5, -0x1.4746100000000p-2, -0x1.48b6720000000p-2}
}
, {{0x1.500d9a0000000p-8, -0x1.13fe0e0000000p-2, -0x1.886d720000000p-5, 0x1.678ae60000000p-2, -0x1.2cc4b40000000p-6, -0x1.9b6d9a0000000p-4, -0x1.d0eff40000000p-3, -0x1.3f39cc0000000p-4, -0x1.50200a0000000p-4, -0x1.0356ec0000000p-2, 0x1.936fcc0000000p-2, 0x1.91ac9e0000000p-3, 0x1.5972a60000000p-3, -0x1.045bd20000000p-1, -0x1.3063540000000p-3, -0x1.5b10200000000p-6}
, {-0x1.b688a40000000p-4, -0x1.147c2e0000000p-2, -0x1.d92d660000000p-3, 0x1.8a51860000000p-2, -0x1.7b737e0000000p-2, -0x1.d024d40000000p-5, -0x1.3322a80000000p-2, -0x1.efc2300000000p-5, -0x1.198d7c0000000p-2, -0x1.0aa7ae0000000p-2, 0x1.3da4200000000p-2, 0x1.5a43ae0000000p-2, 0x1.1caa0a0000000p-2, -0x1.c710980000000p-2, -0x1.3c391c0000000p-3, -0x1.8b0b320000000p-3}
, {-0x1.e5d1500000000p-4, -0x1.c1a8480000000p-3, 0x1.5056fc0000000p-4, 0x1.b698a40000000p-3, 0x1.d3ffde0000000p-5, -0x1.e102660000000p-3, -0x1.d194300000000p-4, 0x1.38c6ba0000000p-6, -0x1.8d8c060000000p-2, -0x1.c2b8a40000000p-4, 0x1.b1bdcc0000000p-3, 0x1.cc38a20000000p-3, -0x1.3102b40000000p-5, -0x1.17f3b00000000p-1, 0x1.4dee3e0000000p-6, -0x1.ccdb140000000p-4}
}
, {{0x1.ac54de0000000p-4, 0x1.9e2b0c0000000p-2, 0x1.9f44d00000000p-2, -0x1.de58cc0000000p-7, 0x1.15765c0000000p-2, 0x1.5ad4280000000p-3, 0x1.23399e0000000p-1, 0x1.bf27ce0000000p-2, 0x1.6776da0000000p-2, 0x1.f4a2520000000p-2, -0x1.eea7ec0000000p-2, 0x1.24f95c0000000p-6, 0x1.6e5a6c0000000p-3, 0x1.7a9b280000000p-2, 0x1.2311ba0000000p-2, -0x1.7a2a5c0000000p-12}
, {0x1.01ef4e0000000p-3, 0x1.86d1be0000000p-2, 0x1.879f500000000p-2, -0x1.ad282e0000000p-3, 0x1.2a87720000000p-1, 0x1.cb613a0000000p-2, 0x1.39638e0000000p-1, 0x1.1b165a0000000p-1, 0x1.840bee0000000p-2, 0x1.3084760000000p-1, -0x1.6daa880000000p-3, -0x1.cec8100000000p-3, -0x1.d79cfa0000000p-6, 0x1.3f39d00000000p-2, 0x1.0644d80000000p-4, -0x1.31352a0000000p-7}
, {0x1.88ab9c0000000p-4, 0x1.cbef180000000p-2, 0x1.70d4880000000p-6, 0x1.56ffc60000000p-7, 0x1.5dd3b60000000p-3, 0x1.1998fe0000000p-1, 0x1.fba00a0000000p-2, 0x1.8540140000000p-2, 0x1.1c58560000000p-1, 0x1.39480a0000000p-2, 0x1.b9b9fc0000000p-4, 0x1.3d481c0000000p-4, 0x1.eb1d6e0000000p-9, 0x1.1b77460000000p-1, 0x1.20f19e0000000p-1, 0x1.21fb280000000p-1}
}
, {{-0x1.7731aa0000000p-4, -0x1.7ddb1e0000000p-3, 0x1.2868280000000p-3, 0x1.528b0e0000000p-2, -0x1.ec278a0000000p-4, -0x1.ac0f440000000p-3, -0x1.7abbd60000000p-3, -0x1.17e0f00000000p-3, -0x1.ac64700000000p-4, -0x1.7a12840000000p-7, 0x1.03d4fc0000000p-1, 0x1.8f6ed20000000p-2, 0x1.7c2ec20000000p-2, -0x1.974f500000000p-2, -0x1.1e4e340000000p-6, -0x1.91d1a60000000p-2}
, {-0x1.06ea4a0000000p-3, -0x1.5ae7020000000p-2, -0x1.157ca40000000p-4, 0x1.cc17b80000000p-2, -0x1.cff2140000000p-4, -0x1.7a53200000000p-5, -0x1.96f2420000000p-6, -0x1.7393480000000p-2, -0x1.b9a91e0000000p-3, -0x1.cd75940000000p-3, 0x1.5a742c0000000p-1, 0x1.9ee28a0000000p-3, 0x1.c68cea0000000p-2, -0x1.f1523c0000000p-2, -0x1.77cd180000000p-2, -0x1.43b86e0000000p-1}
, {-0x1.8ff78a0000000p-3, -0x1.f6e45a0000000p-3, 0x1.2a62de0000000p-5, 0x1.f557a40000000p-3, -0x1.2a83320000000p-3, -0x1.029b180000000p-3, -0x1.f42a160000000p-5, -0x1.a690500000000p-3, -0x1.e1d9840000000p-5, 0x1.640dd80000000p-4, 0x1.2c9ffc0000000p-1, 0x1.3f2e100000000p-3, 0x1.1ba5100000000p-4, -0x1.a975c60000000p-2, -0x1.d3642a0000000p-3, -0x1.3b91ea0000000p-1}
}
, {{0x1.107ca40000000p+0, 0x1.f48f740000000p-2, 0x1.16c7e00000000p-3, 0x1.25d89c0000000p-1, 0x1.0492280000000p-3, 0x1.88633c0000000p-2, 0x1.bdf5f60000000p-3, 0x1.0c049c0000000p-5, 0x1.3fa1b40000000p-2, 0x1.4de1a20000000p-2, 0x1.1525820000000p-3, 0x1.51dc5c0000000p-3, 0x1.44179a0000000p-3, 0x1.d2b2560000000p-3, 0x1.41d3120000000p-1, 0x1.5694c00000000p-1}
, {0x1.9f8e800000000p-1, 0x1.ae4b960000000p-4, -0x1.cde5540000000p-4, 0x1.456e5a0000000p-1, -0x1.25911a0000000p-4, -0x1.3186d20000000p-2, -0x1.b454320000000p-4, -0x1.237b400000000p-3, -0x1.051d9c0000000p-3, -0x1.8a23560000000p-3, 0x1.0bc8980000000p-2, 0x1.4862d60000000p-2, 0x1.3785f00000000p-2, 0x1.4457e80000000p-7, 0x1.5cbb380000000p-2, 0x1.f278d40000000p-1}
, {0x1.01cf580000000p-1, -0x1.54fb320000000p-2, -0x1.34834e0000000p-1, 0x1.3367020000000p-1, -0x1.0704ce0000000p-1, -0x1.2e0ace0000000p-2, -0x1.32f3500000000p-2, -0x1.91adae0000000p-2, -0x1.904ad00000000p-2, -0x1.3551380000000p-2, 0x1.6c99860000000p-4, 0x1.6e378c0000000p-2, 0x1.7dbc100000000p-2, -0x1.674ea20000000p-2, 0x1.da7c340000000p-2, 0x1.c3b54e0000000p-1}
}
, {{-0x1.ea59a60000000p-3, -0x1.b64e320000000p-6, 0x1.d0e7f60000000p-3, 0x1.9e4d1c0000000p-2, -0x1.93468e0000000p-3, 0x1.78beb00000000p-6, 0x1.3daef40000000p-3, 0x1.c4cd460000000p-6, -0x1.32378c0000000p-3, -0x1.3da2280000000p-4, 0x1.27bd1c0000000p-1, 0x1.2571de0000000p-2, 0x1.1b684c0000000p-2, -0x1.179bd80000000p-4, -0x1.af788e0000000p-4, -0x1.d4d5840000000p-9}
, {0x1.44059c0000000p-4, 0x1.26bbce0000000p-4, -0x1.1b2b320000000p-5, 0x1.d4fa780000000p-2, 0x1.e6e6620000000p-5, -0x1.8ac1a20000000p-4, -0x1.1b93040000000p-3, 0x1.4ebfb20000000p-4, 0x1.36c0bc0000000p-4, -0x1.26f4be0000000p-3, 0x1.0750300000000p-1, 0x1.36816c0000000p-2, 0x1.1c256a0000000p-2, -0x1.eed8140000000p-4, -0x1.eff6bc0000000p-3, -0x1.6cf1ea0000000p-2}
, {-0x1.5119020000000p-4, 0x1.22f08a0000000p-9, -0x1.46bc160000000p-3, 0x1.d146ae0000000p-2, -0x1.01590a0000000p-4, 0x1.42e58a0000000p-3, 0x1.39ac8e0000000p-6, 0x1.afc6ac0000000p-5, -0x1.b43d080000000p-6, -0x1.ad77380000000p-4, 0x1.08754e0000000p-1, 0x1.90d5740000000p-4, 0x1.3943a40000000p-2, 0x1.2d20060000000p-10, -0x1.bb76320000000p-4, -0x1.7c7cd80000000p-3}
}
, {{0x1.8406da0000000p-1, -0x1.2392060000000p-5, -0x1.e48b9c0000000p-2, 0x1.9529140000000p-3, -0x1.f01aee0000000p-3, -0x1.6ae1880000000p-4, -0x1.541baa0000000p-3, -0x1.45fa960000000p-2, -0x1.1ba5320000000p-3, -0x1.581d640000000p-2, 0x1.73fbf20000000p-3, 0x1.61eabe0000000p-5, 0x1.3c4bfc0000000p-2, -0x1.15fdea0000000p-5, 0x1.268a4a0000000p-4, 0x1.44cd6c0000000p-1}
, {0x1.01ba840000000p-1, -0x1.30d8420000000p-3, -0x1.b6a10e0000000p-2, 0x1.1f44740000000p-3, -0x1.d15e320000000p-2, -0x1.f21f520000000p-2, -0x1.6fa21a0000000p-2, -0x1.24a8fc0000000p-2, -0x1.1272c20000000p-3, -0x1.0466200000000p-1, -0x1.2439f40000000p-6, 0x1.b9d6340000000p-3, 0x1.4bb8420000000p-2, -0x1.05ed020000000p-7, 0x1.a92eb40000000p-2, 0x1.82c7760000000p-1}
, {0x1.acac4e0000000p-2, 0x1.cbff060000000p-3, -0x1.7998da0000000p-2, 0x1.f19e3a0000000p-4, 0x1.14ba940000000p-5, -0x1.6ed29e0000000p-3, 0x1.d37e8c0000000p-4, -0x1.349cb40000000p-4, -0x1.d961b40000000p-3, -0x1.3aedb80000000p-2, 0x1.0f79c00000000p-4, 0x1.36bd980000000p-2, 0x1.c599640000000p-3, 0x1.328a220000000p-4, 0x1.2708b40000000p-1, 0x1.eb93020000000p-1}
}
, {{0x1.cc3ed20000000p-1, 0x1.3e650a0000000p-3, -0x1.72c4f80000000p-2, 0x1.ec58100000000p-2, -0x1.2f02220000000p-1, -0x1.6e76e20000000p-2, -0x1.77e5860000000p-6, -0x1.f864c80000000p-2, -0x1.77052c0000000p-2, -0x1.f755120000000p-2, 0x1.6dabb60000000p-2, 0x1.5f34360000000p-2, 0x1.2e04080000000p-4, -0x1.2746160000000p-1, 0x1.1145be0000000p-3, 0x1.bc69460000000p-1}
, {0x1.360a4c0000000p-1, -0x1.c37fe60000000p-3, -0x1.a9c7060000000p-2, 0x1.85a44e0000000p-3, -0x1.be22ec0000000p-1, -0x1.eea0fe0000000p-3, -0x1.db028a0000000p-2, -0x1.4be5b60000000p-1, -0x1.29c57e0000000p-1, -0x1.51d14a0000000p-2, 0x1.9feca40000000p-2, 0x1.01089c0000000p-3, 0x1.d8e84e0000000p-5, -0x1.3a03e00000000p-1, 0x1.8353ae0000000p-1, 0x1.2539340000000p+0}
, {0x1.7e7ce60000000p-1, 0x1.63b98a0000000p-3, -0x1.cf25ee0000000p-5, 0x1.36db0a0000000p-2, -0x1.0862e60000000p-1, 0x1.c2ce0e0000000p-3, 0x1.b2c1000000000p-3, -0x1.8f0a2a0000000p-5, -0x1.0038de0000000p-2, -0x1.4469b80000000p-2, 0x1.44cede0000000p-3, 0x1.2355140000000p-3, 0x1.1153840000000p-2, -0x1.c254260000000p-3, 0x1.e5d3fa0000000p-2, 0x1.23f4a40000000p+0}
}
, {{0x1.2c27820000000p-1, 0x1.38e4580000000p-2, 0x1.4d392c0000000p-4, 0x1.51f2c80000000p-3, 0x1.ef91ea0000000p-10, 0x1.30cd1c0000000p-2, 0x1.ebdade0000000p-2, 0x1.229bbe0000000p-3, 0x1.3264b80000000p-2, 0x1.76500c0000000p-4, -0x1.f9702c0000000p-3, -0x1.f01b680000000p-4, -0x1.5fae140000000p-4, 0x1.5dc6a00000000p-2, 0x1.58ddd60000000p-2, 0x1.524c0e0000000p-1}
, {0x1.a10c080000000p-2, 0x1.5fdbb60000000p-2, 0x1.09411a0000000p-3, -0x1.148eb80000000p-10, 0x1.4371ae0000000p-3, -0x1.3334b00000000p-5, -0x1.2392480000000p-4, 0x1.7665ec0000000p-3, 0x1.165ed60000000p-2, -0x1.838c280000000p-6, -0x1.1672a40000000p-4, -0x1.e1d22a0000000p-5, 0x1.a556ce0000000p-3, -0x1.8ac0380000000p-5, 0x1.3facc60000000p-3, 0x1.5d50960000000p-3}
, {0x1.cb75a40000000p-3, 0x1.c1c49e0000000p-2, 0x1.2205d80000000p-2, 0x1.1a634a0000000p-7, 0x1.0366de0000000p-4, 0x1.61bf800000000p-2, 0x1.2148960000000p-1, 0x1.fcf50a0000000p-2, 0x1.2da9160000000p-2, 0x1.5016680000000p-2, 0x1.a75cf20000000p-3, -0x1.de281e0000000p-3, -0x1.2fb3e20000000p-3, 0x1.0159a20000000p-1, 0x1.ff4a700000000p-4, 0x1.cc4d280000000p-1}
}
, {{0x1.30b5240000000p+0, 0x1.cde3f60000000p-2, 0x1.05c1000000000p-2, 0x1.0684b60000000p-1, 0x1.e246440000000p-3, 0x1.7247140000000p-3, 0x1.1de4e40000000p-2, 0x1.acde800000000p-2, 0x1.cc27fc0000000p-3, 0x1.b7a8fe0000000p-4, -0x1.844c4e0000000p-4, -0x1.c8cbc00000000p-5, 0x1.38cc920000000p-3, 0x1.5a91f20000000p-3, 0x1.a8cae40000000p-3, 0x1.049c400000000p-1}
, {0x1.560e960000000p-1, 0x1.23e2a00000000p-2, 0x1.ba3e4e0000000p-5, 0x1.a966780000000p-3, -0x1.81cde40000000p-5, 0x1.4807a60000000p-2, 0x1.ac7e640000000p-5, -0x1.0de0940000000p-5, 0x1.060a300000000p-4, 0x1.7ee5860000000p-6, 0x1.75caf00000000p-5, -0x1.779ae60000000p-3, -0x1.792f720000000p-5, 0x1.156dd60000000p-2, 0x1.d974660000000p-2, 0x1.d6366a0000000p-1}
, {0x1.0d83240000000p-1, 0x1.f6089a0000000p-2, 0x1.a4fb3e0000000p-3, 0x1.1cce900000000p-3, 0x1.bd91600000000p-3, 0x1.d1b18a0000000p-2, 0x1.2340800000000p-1, 0x1.a8da660000000p-2, 0x1.bc002e0000000p-2, 0x1.c6ed540000000p-3, 0x1.0a728c0000000p-2, -0x1.a94ed00000000p-6, 0x1.a3bc3a0000000p-4, 0x1.813fa60000000p-3, 0x1.69260e0000000p-3, 0x1.7458b00000000p-1}
}
, {{0x1.0e88820000000p-1, -0x1.c0add80000000p-5, 0x1.dced240000000p-3, 0x1.ad41ec0000000p-2, 0x1.11ffcc0000000p-4, 0x1.a79d000000000p-11, 0x1.f6e4ec0000000p-3, -0x1.5b8a0a0000000p-6, -0x1.5f583c0000000p-5, 0x1.ccaeda0000000p-5, -0x1.b696c20000000p-6, 0x1.5b3b860000000p-3, -0x1.4a4cfc0000000p-6, 0x1.aa55b60000000p-7, 0x1.45cc420000000p-1, 0x1.4bb8f80000000p-2}
, {0x1.3120840000000p-2, 0x1.e0116a0000000p-2, 0x1.aa82b60000000p-2, -0x1.9ddd2e0000000p-3, 0x1.bf28e20000000p-2, 0x1.3f99b20000000p-2, 0x1.27fd480000000p-2, 0x1.027ea20000000p-1, 0x1.cb87220000000p-3, 0x1.f7fb940000000p-2, -0x1.bb3ff40000000p-2, -0x1.90e1e60000000p-3, 0x1.e305880000000p-4, 0x1.929a420000000p-3, 0x1.f838f20000000p-4, -0x1.1396b80000000p-5}
, {-0x1.b2d7a40000000p-3, 0x1.9b5f460000000p-2, 0x1.9977440000000p-2, -0x1.aa3a8e0000000p-2, 0x1.06767a0000000p-2, 0x1.b443ca0000000p-2, 0x1.bf6ac20000000p-2, 0x1.09a9c00000000p-1, 0x1.50caca0000000p-2, 0x1.10fb2a0000000p-1, -0x1.f5270e0000000p-2, -0x1.d452880000000p-4, -0x1.18e5780000000p-3, 0x1.c70db00000000p-2, 0x1.7df2120000000p-1, 0x1.c7df5a0000000p-1}
}
, {{0x1.6ba9e80000000p+0, 0x1.0b46960000000p-1, 0x1.0363f20000000p-5, 0x1.e959ce0000000p-2, 0x1.69685c0000000p-6, 0x1.a80d260000000p-3, 0x1.a418ec0000000p-3, 0x1.416f520000000p-2, -0x1.eed5c60000000p-5, 0x1.efc16c0000000p-4, 0x1.81ba360000000p-4, -0x1.d9e3ce0000000p-4, 0x1.5e9ac80000000p-3, -0x1.acdbc60000000p-7, 0x1.927f6e0000000p-6, 0x1.72e61a0000000p-2}
, {0x1.ba48a80000000p-2, 0x1.59be020000000p-5, -0x1.6e46d00000000p-6, 0x1.e9eeae0000000p-3, -0x1.dba40c0000000p-4, -0x1.5c8a2a0000000p-3, -0x1.48183a0000000p-4, 0x1.e1d8ac0000000p-7, -0x1.493bf00000000p-3, -0x1.06b2260000000p-4, 0x1.9414220000000p-7, 0x1.9130f40000000p-6, -0x1.06f2780000000p-6, 0x1.1bf7360000000p-4, 0x1.5089480000000p-1, 0x1.aec47e0000000p-1}
, {0x1.2c9b9c0000000p-1, 0x1.2943380000000p-1, 0x1.aa40a40000000p-9, 0x1.a6916e0000000p-4, 0x1.3769b80000000p-2, 0x1.1d49520000000p-2, 0x1.01fe580000000p-1, 0x1.3cb5e60000000p-2, 0x1.e77d600000000p-3, 0x1.bcd5400000000p-2, -0x1.a9a2260000000p-6, 0x1.37c84a0000000p-3, 0x1.e9efd40000000p-4, 0x1.9309180000000p-5, 0x1.77b2d40000000p-1, 0x1.1a0a180000000p+0}
}
, {{0x1.b8f9ca0000000p-1, 0x1.3e713e0000000p-4, 0x1.73aa880000000p-5, 0x1.aaa7400000000p-3, -0x1.0a6cc00000000p-3, 0x1.e849ec0000000p-5, 0x1.1bd7c80000000p-2, 0x1.615d100000000p-3, -0x1.6a8a600000000p-5, 0x1.19e30a0000000p-4, 0x1.711ae60000000p-2, 0x1.7a4a8c0000000p-6, 0x1.6a88d40000000p-2, -0x1.a50af00000000p-4, 0x1.125a120000000p-2, 0x1.1b151a0000000p-3}
, {0x1.22e3c80000000p-2, -0x1.441c2e0000000p-4, -0x1.e1cb800000000p-3, 0x1.6ad1ba0000000p-2, -0x1.94a6d00000000p-3, -0x1.d0046a0000000p-4, 0x1.be2a660000000p-4, 0x1.7e6ed00000000p-5, -0x1.3fb6360000000p-3, 0x1.7609bc0000000p-4, -0x1.edc7a20000000p-5, 0x1.7e24220000000p-4, 0x1.76e7840000000p-4, 0x1.84ee0c0000000p-3, 0x1.2ba0ea0000000p-2, 0x1.5ea5be0000000p-1}
, {0x1.6a68fa0000000p-1, 0x1.20aba60000000p-2, 0x1.61ca3a0000000p-3, 0x1.59dfa80000000p-2, 0x1.b747fe0000000p-4, 0x1.8eb5f80000000p-3, -0x1.580eaa0000000p-4, 0x1.7bc5e60000000p-3, -0x1.d7786c0000000p-9, -0x1.524aa00000000p-5, 0x1.ded2620000000p-3, -0x1.81aad60000000p-4, 0x1.d0f7480000000p-3, -0x1.0351d40000000p-3, 0x1.4878500000000p-1, 0x1.36bb9c0000000p-1}
}
, {{0x1.9b9d660000000p-4, -0x1.3602d20000000p-3, 0x1.2630840000000p-4, 0x1.e697100000000p-3, 0x1.75fbd80000000p-3, -0x1.4d4f3c0000000p-4, 0x1.b42f020000000p-3, 0x1.92890c0000000p-3, 0x1.de0ffa0000000p-6, 0x1.94a2ce0000000p-2, 0x1.735a000000000p-2, -0x1.0946620000000p-2, 0x1.2116220000000p-5, -0x1.6b3bb60000000p-3, -0x1.0e8edc0000000p-3, -0x1.4e35ba0000000p-1}
, {-0x1.c8d9420000000p-5, 0x1.62841c0000000p-4, 0x1.77d00c0000000p-2, -0x1.e7d15a0000000p-2, 0x1.39c8ac0000000p-1, 0x1.58bda80000000p-2, 0x1.08f4b00000000p-2, 0x1.d6ce460000000p-2, 0x1.3b51380000000p-4, 0x1.e83d8e0000000p-3, -0x1.85f55a0000000p-2, -0x1.7d11440000000p-4, -0x1.75cd600000000p-2, 0x1.d63e6e0000000p-5, -0x1.9e8e600000000p-2, -0x1.656bf60000000p-1}
, {-0x1.61513a0000000p-1, 0x1.8386f80000000p-2, 0x1.d2ad360000000p-3, -0x1.6afc0e0000000p-2, 0x1.e805b40000000p-2, 0x1.8fce340000000p-2, 0x1.5177ea0000000p-2, 0x1.f2d3d60000000p-3, 0x1.00722e0000000p-3, 0x1.695de80000000p-2, 0x1.81af100000000p-2, 0x1.614e120000000p-4, -0x1.ff765a0000000p-3, 0x1.8727280000000p-4, 0x1.a2f0120000000p-1, 0x1.c215e00000000p-2}
}
, {{0x1.63a74e0000000p+0, 0x1.cdc66a0000000p-2, 0x1.310b680000000p-6, 0x1.4f97440000000p-2, 0x1.38918e0000000p-6, 0x1.987f940000000p-3, 0x1.53f9d80000000p-3, 0x1.53d7760000000p-2, 0x1.fb21cc0000000p-3, 0x1.67c2000000000p-3, 0x1.ac5bd20000000p-3, -0x1.f17ebe0000000p-5, 0x1.0de7f80000000p-2, 0x1.2334320000000p-4, 0x1.e1540e0000000p-3, 0x1.5cfa360000000p-3}
, {0x1.7453720000000p-1, -0x1.fe8c880000000p-5, 0x1.c356b20000000p-4, 0x1.ab9d0a0000000p-2, -0x1.1bdab60000000p-5, 0x1.5340820000000p-3, 0x1.0a34260000000p-3, -0x1.0fe9280000000p-3, -0x1.3ffaca0000000p-6, -0x1.34f54c0000000p-3, 0x1.113a100000000p-3, -0x1.3ed5fa0000000p-3, 0x1.6de34c0000000p-3, 0x1.90667a0000000p-3, 0x1.5b987e0000000p-2, 0x1.ab82f40000000p-1}
, {0x1.9b0ac20000000p-2, 0x1.abc42c0000000p-3, 0x1.0358320000000p-2, 0x1.61be100000000p-2, 0x1.9080d00000000p-3, 0x1.57cb420000000p-2, 0x1.772abe0000000p-2, 0x1.98fc000000000p-2, 0x1.28e0bc0000000p-2, 0x1.86bf280000000p-2, 0x1.2430dc0000000p-4, 0x1.ca682a0000000p-5, 0x1.d4c16e0000000p-3, 0x1.78c7960000000p-4, 0x1.31fa9e0000000p-1, 0x1.456b600000000p+0}
}
, {{-0x1.6ed4340000000p-3, 0x1.82a9ec0000000p-2, 0x1.86f80c0000000p-2, -0x1.8b32aa0000000p-4, 0x1.c706ba0000000p-3, 0x1.0ce0f00000000p-1, 0x1.bdce7c0000000p-2, 0x1.c4eee60000000p-2, 0x1.2daa6c0000000p-2, 0x1.0e1f160000000p-1, -0x1.40567c0000000p-2, -0x1.d707160000000p-4, -0x1.d530780000000p-4, 0x1.ecd1e60000000p-2, 0x1.8cbd980000000p-5, -0x1.46a80a0000000p-1}
, {-0x1.1811b80000000p-1, 0x1.b69cf80000000p-4, 0x1.21d8160000000p-3, -0x1.7f7a1c0000000p-2, 0x1.5f6f4e0000000p-2, 0x1.c78b4a0000000p-2, 0x1.04df0e0000000p-1, 0x1.20bec80000000p-1, 0x1.02c1940000000p-2, 0x1.67ca440000000p-2, -0x1.ceba6a0000000p-3, -0x1.24614a0000000p-6, -0x1.745b380000000p-2, 0x1.a49af00000000p-3, 0x1.c03f0c0000000p-2, 0x1.04d2f60000000p-1}
, {-0x1.450f8c0000000p-3, 0x1.3ff7540000000p-2, 0x1.7941900000000p-2, 0x1.feffd60000000p-2, 0x1.9ed7ac0000000p-4, 0x1.4b477c0000000p-4, 0x1.325b200000000p-3, 0x1.5432ea0000000p-4, 0x1.2a5c300000000p-5, 0x1.abb9260000000p-5, 0x1.4132920000000p-2, -0x1.db4d800000000p-4, -0x1.9a4b060000000p-6, 0x1.4787060000000p-4, 0x1.55d6220000000p-1, 0x1.59ac6e0000000p-1}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_2_H_
#define _MAX_POOLING1D_2_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  30
#define INPUT_SAMPLES   330
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef float max_pooling1d_2_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_2(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_2_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_2.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  30
#define INPUT_SAMPLES   330
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void max_pooling1d_2(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#else
#error "Unsupported activation function"
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    averagepool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _AVERAGE_POOLING1D_H_
#define _AVERAGE_POOLING1D_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  30
#define INPUT_SAMPLES   82
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef float average_pooling1d_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void average_pooling1d(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_AVERAGE_POOLING1D_H_
/**
  ******************************************************************************
  * @file    averagepool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "average_pooling1d.h"
#include "number.h"
#endif

#define INPUT_CHANNELS  30
#define INPUT_SAMPLES   82
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void average_pooling1d(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  LONG_NUMBER_T avg, tmp;

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
      tmp = 0;
      for (x = 0; x < POOL_SIZE; x++) {
        tmp += input[(pos_x*POOL_STRIDE)+x][k];
      }
#ifdef ACTIVATION_RELU
      if (tmp < 0) {
        tmp = 0;
      }
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation function"
#endif
      avg = tmp / POOL_SIZE;

      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, avg, INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_2_H_
#define _CONV1D_2_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      30
#define INPUT_SAMPLES       41
#define CONV_FILTERS        4
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef float conv1d_2_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_2(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_2_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_2.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      30
#define INPUT_SAMPLES       41
#define CONV_FILTERS        4
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 0
#define BIASES_SCALE_FACTOR 0
#define TMP_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void conv1d_2(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

#error "Data type unsupported by CMSIS-NN"

#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    30
#define CONV_FILTERS      4
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const float  conv1d_2_bias[CONV_FILTERS] = {0x1.6b912e0000000p-4, 0x1.ece74e0000000p-6, -0x1.933b620000000p-3, 0x1.9954540000000p-4}
;

const float  conv1d_2_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{0x1.2c4d160000000p-2, 0x1.95891c0000000p-4, 0x1.7166840000000p-2, -0x1.0230860000000p-2, 0x1.4b09040000000p-2, -0x1.14e49c0000000p-10, -0x1.54d9bc0000000p-5, -0x1.1ff9060000000p-2, 0x1.8f22c20000000p-3, -0x1.029f8e0000000p-2, -0x1.63610c0000000p-1, 0x1.4962c60000000p-5, -0x1.24e7b00000000p-3, 0x1.88a6bc0000000p-3, -0x1.5cc39e0000000p-1, -0x1.36e8be0000000p-2, 0x1.0c3a980000000p-3, -0x1.10bdc00000000p-1, -0x1.e5e6100000000p-5, -0x1.7f18320000000p-3, 0x1.3161c00000000p-2, 0x1.79a48e0000000p-2, 0x1.3947660000000p-2, 0x1.bcccba0000000p-4, 0x1.5ddb900000000p-10, 0x1.53a6a40000000p-2, 0x1.c70ab20000000p-3, 0x1.668a580000000p-3, 0x1.4f92ea0000000p-4, -0x1.a1f71c0000000p-6}
, {-0x1.3543460000000p-4, -0x1.6ed83e0000000p-3, -0x1.a75a580000000p-6, -0x1.5649ac0000000p-3, -0x1.5882400000000p-6, 0x1.5b61b60000000p-3, 0x1.d750800000000p-3, -0x1.fffa560000000p-4, 0x1.9cd7e60000000p-3, 0x1.4e7cb40000000p-6, 0x1.4ea92c0000000p-4, 0x1.892f860000000p-3, 0x1.67b5320000000p-5, -0x1.86ef1e0000000p-3, -0x1.679b0c0000000p-4, -0x1.7414000000000p-4, 0x1.f023fa0000000p-5, -0x1.c644ce0000000p-3, 0x1.ac06f60000000p-5, -0x1.07e1e80000000p-4, 0x1.f0abde0000000p-3, 0x1.09750e0000000p-2, -0x1.867f840000000p-3, 0x1.a640140000000p-4, 0x1.8154440000000p-5, 0x1.d31b1e0000000p-4, 0x1.4efabe0000000p-3, -0x1.cf6a9a0000000p-3, -0x1.6475160000000p-4, 0x1.53efd40000000p-3}
, {-0x1.48674e0000000p-3, -0x1.2ca2520000000p-1, -0x1.080b300000000p-1, 0x1.b0c94c0000000p-6, -0x1.7a1fd80000000p-3, -0x1.68cbd60000000p-3, 0x1.a667e20000000p-6, 0x1.0cf4c40000000p-3, -0x1.e8227c0000000p-4, -0x1.0dc6a40000000p-3, 0x1.3acbd60000000p-4, -0x1.5fa9c00000000p-5, 0x1.ecfc6a0000000p-4, -0x1.bf547e0000000p-2, -0x1.1a3d100000000p-3, 0x1.1e49b40000000p-2, -0x1.0473740000000p-1, -0x1.d7d4820000000p-5, 0x1.97af320000000p-5, -0x1.33299e0000000p-3, -0x1.8caa400000000p-4, 0x1.3e6c1e0000000p-2, -0x1.ee97940000000p-2, -0x1.0750000000000p-2, -0x1.c248f80000000p-3, -0x1.6dca320000000p-7, -0x1.33ca920000000p-2, -0x1.9525640000000p-2, -0x1.950e760000000p-5, -0x1.e558bc0000000p-2}
}
, {{-0x1.0f5e740000000p-3, 0x1.806b700000000p-3, 0x1.e6c1b40000000p-3, -0x1.af58440000000p-3, 0x1.f6633a0000000p-2, 0x1.dbca120000000p-3, -0x1.30e94e0000000p-5, 0x1.2c788a0000000p-2, 0x1.34d1060000000p-9, -0x1.109e7e0000000p-2, 0x1.0353180000000p-4, 0x1.5f20740000000p-3, 0x1.3547c60000000p-5, 0x1.4e5a5a0000000p-3, -0x1.a7c1380000000p-6, 0x1.43c1560000000p-4, 0x1.db81040000000p-2, 0x1.1b3b3e0000000p-3, 0x1.0a04760000000p-2, 0x1.fc603c0000000p-3, -0x1.0997d00000000p-3, 0x1.3153e20000000p-3, 0x1.dbb9440000000p-2, 0x1.5675300000000p-6, 0x1.2020de0000000p-1, -0x1.afcfc80000000p-4, -0x1.f1db580000000p-2, 0x1.2993380000000p-1, -0x1.18fa8a0000000p-2, 0x1.4e43980000000p-2}
, {0x1.b24e500000000p-5, 0x1.0ac1ca0000000p-2, 0x1.8f1d0e0000000p-3, 0x1.b3584c0000000p-2, 0x1.0cc15c0000000p-4, 0x1.77422a0000000p-3, 0x1.9fe2ec0000000p-2, 0x1.0cf5c20000000p-2, -0x1.b9137a0000000p-5, 0x1.d501560000000p-2, 0x1.fa62bc0000000p-3, 0x1.3fc00c0000000p-3, 0x1.69a5640000000p-4, 0x1.9059240000000p-3, 0x1.193e8c0000000p-5, 0x1.8282420000000p-2, 0x1.4849420000000p-7, 0x1.04642e0000000p-1, 0x1.627b880000000p-1, 0x1.bb78060000000p-2, 0x1.cedada0000000p-2, 0x1.1ee7f80000000p-1, 0x1.13e7b40000000p-4, -0x1.5a62220000000p-3, 0x1.3852b00000000p-2, -0x1.5302b80000000p-3, -0x1.5baba40000000p-3, 0x1.7bd1680000000p-3, 0x1.a58bb40000000p-9, 0x1.71f6820000000p-4}
, {0x1.9a696e0000000p-5, 0x1.6472760000000p-2, 0x1.d15c520000000p-2, -0x1.ad15da0000000p-4, 0x1.c82b500000000p-2, -0x1.d078fc0000000p-3, -0x1.7de38a0000000p-2, 0x1.18319c0000000p-2, 0x1.1412d40000000p-3, -0x1.08f6500000000p-2, -0x1.de7f340000000p-5, -0x1.cbdeda0000000p-4, -0x1.c846e20000000p-3, 0x1.4005960000000p-3, -0x1.b9a4360000000p-11, 0x1.35c9ae0000000p-3, 0x1.38be9c0000000p-3, 0x1.7f228e0000000p-11, 0x1.767d340000000p-12, 0x1.c3bd940000000p-3, 0x1.962f200000000p-3, 0x1.36b06e0000000p-4, -0x1.367b8a0000000p-6, 0x1.74fed60000000p-3, 0x1.eb4f300000000p-2, -0x1.6bc8e00000000p-2, -0x1.2de3280000000p-1, 0x1.d1a5900000000p-2, -0x1.0c66200000000p-2, 0x1.c51ce00000000p-2}
}
, {{0x1.34613e0000000p-1, -0x1.0fe7600000000p-6, -0x1.7515ae0000000p-3, 0x1.9a612c0000000p-12, 0x1.3da9dc0000000p-5, -0x1.fdb1840000000p-5, -0x1.9216680000000p-6, -0x1.3e532e0000000p-6, -0x1.fe16700000000p-4, 0x1.e266300000000p-5, -0x1.722d560000000p-6, 0x1.9b1c9a0000000p-2, -0x1.7bf92e0000000p-3, -0x1.024b700000000p-2, 0x1.6b70a60000000p-3, -0x1.daca480000000p-7, -0x1.52bb240000000p-2, -0x1.45f9b60000000p-2, 0x1.ec911a0000000p-3, -0x1.d835e00000000p-3, -0x1.29473e0000000p-2, 0x1.048d3e0000000p-3, 0x1.60b8c00000000p-3, 0x1.1090a80000000p-3, -0x1.a24a8c0000000p-5, 0x1.91c53a0000000p-1, -0x1.9afc740000000p-4, -0x1.7b73c40000000p-4, 0x1.fb02680000000p-3, -0x1.3ce5aa0000000p-2}
, {0x1.98159e0000000p-2, 0x1.3038860000000p-2, 0x1.3ebf980000000p-3, -0x1.c0da520000000p-2, 0x1.34fa2c0000000p-2, 0x1.7327b40000000p-4, 0x1.f8e4b40000000p-3, -0x1.7b31f00000000p-2, 0x1.29feae0000000p-3, 0x1.7ce4120000000p-1, -0x1.b9fa5e0000000p-2, 0x1.9eb2d00000000p-1, -0x1.800c500000000p-4, 0x1.bb3eb20000000p-2, -0x1.c9faa60000000p-2, -0x1.95df5e0000000p-2, 0x1.55e76a0000000p-2, -0x1.e0b85a0000000p-2, 0x1.1574940000000p-2, -0x1.e9ce0e0000000p-3, 0x1.ec44360000000p-4, -0x1.15e3c80000000p-2, 0x1.4e4a3a0000000p-2, 0x1.2cf7ba0000000p-1, 0x1.e798500000000p-3, 0x1.cd518c0000000p-1, -0x1.3bad2c0000000p-4, -0x1.1eee7c0000000p-3, 0x1.8a27320000000p-1, 0x1.21ef480000000p-6}
, {0x1.55e1060000000p-1, -0x1.52761a0000000p-3, 0x1.bbbfce0000000p-2, -0x1.1f1d280000000p-2, 0x1.dce00a0000000p-3, 0x1.c287ba0000000p-5, 0x1.9239f60000000p-2, 0x1.15b5d80000000p-5, -0x1.3e10220000000p-3, 0x1.2b3ce20000000p-1, -0x1.cf28400000000p-4, 0x1.5e1e320000000p-1, -0x1.15e5040000000p-5, 0x1.a1f1380000000p-4, -0x1.b8aa920000000p-3, -0x1.8d08520000000p-6, 0x1.c4821c0000000p-3, -0x1.496ac60000000p-3, 0x1.02cdfc0000000p-1, -0x1.3def960000000p-2, 0x1.10da580000000p-6, -0x1.c0e4260000000p-3, 0x1.5e39d60000000p-2, 0x1.23e3940000000p-1, 0x1.8cc80a0000000p-4, 0x1.086c600000000p-1, -0x1.19e37a0000000p-3, -0x1.d986d60000000p-3, 0x1.5049cc0000000p-2, 0x1.941b320000000p-3}
}
, {{0x1.9f3d540000000p-2, -0x1.1df4de0000000p-4, 0x1.c296a80000000p-7, 0x1.60a71c0000000p-3, 0x1.ff77420000000p-4, -0x1.0c86ba0000000p-3, 0x1.610bb60000000p-2, -0x1.043f0e0000000p-4, -0x1.de9cd80000000p-5, 0x1.1f63ee0000000p-4, -0x1.fa7a980000000p-5, 0x1.76477e0000000p-2, 0x1.c9b0760000000p-3, -0x1.44784e0000000p-2, -0x1.b20a240000000p-4, 0x1.e338320000000p-5, -0x1.2fb02e0000000p-2, -0x1.a279fe0000000p-6, -0x1.8afafc0000000p-3, 0x1.b860e80000000p-3, 0x1.536b160000000p-3, -0x1.1d66cc0000000p-5, 0x1.febfc40000000p-3, 0x1.4910520000000p-1, -0x1.cee0b20000000p-3, 0x1.850fee0000000p-2, 0x1.97ddb20000000p-4, -0x1.0e8a6e0000000p-3, 0x1.0b76e80000000p-2, 0x1.64bb340000000p-5}
, {0x1.87bdf80000000p-2, -0x1.e23bd80000000p-2, 0x1.4020860000000p-2, 0x1.aab4360000000p-2, 0x1.48d2220000000p-7, 0x1.07101c0000000p-4, 0x1.7fbf6a0000000p-1, 0x1.9851060000000p-2, -0x1.939c400000000p-3, 0x1.2a63100000000p-2, 0x1.a0c8c60000000p-2, 0x1.28abb00000000p-2, 0x1.a557ca0000000p-4, 0x1.c1975e0000000p-5, 0x1.1b05820000000p-1, 0x1.38e21e0000000p-2, -0x1.08b8b80000000p-3, 0x1.9421a00000000p-2, 0x1.36d45c0000000p-2, 0x1.40e5f60000000p-2, 0x1.a244c20000000p-2, 0x1.83ee660000000p-2, 0x1.0cdce60000000p-5, 0x1.4297540000000p-1, -0x1.337bb00000000p-2, 0x1.ea6bc60000000p-2, 0x1.aa99dc0000000p-2, -0x1.ff65760000000p-3, 0x1.4c914e0000000p-1, -0x1.1aa2480000000p-3}
, {0x1.47c5360000000p-4, -0x1.8f0a5e0000000p-2, 0x1.8d8fde0000000p-2, 0x1.bd87ba0000000p-4, -0x1.c57bc40000000p-3, 0x1.78b2fa0000000p-4, 0x1.7ad25e0000000p-2, -0x1.ad82fe0000000p-4, -0x1.c8db900000000p-5, 0x1.143cea0000000p-5, 0x1.77c1ca0000000p-6, 0x1.2198300000000p-4, -0x1.58248e0000000p-3, -0x1.6359100000000p-3, 0x1.0060720000000p-2, 0x1.c2ca9e0000000p-4, -0x1.5af9580000000p-2, 0x1.f16bf20000000p-3, -0x1.e5b3620000000p-3, 0x1.2ff2660000000p-2, 0x1.060b7e0000000p-4, 0x1.504b300000000p-3, 0x1.2cdfa00000000p-2, 0x1.ee782c0000000p-2, -0x1.41aa3c0000000p-2, 0x1.84eb780000000p-2, 0x1.754ec80000000p-4, 0x1.64331e0000000p-4, 0x1.454d800000000p-4, -0x1.2c9d8c0000000p-3}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_3_H_
#define _MAX_POOLING1D_3_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  4
#define INPUT_SAMPLES   39
#define POOL_SIZE       8
#define POOL_STRIDE     8
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef float max_pooling1d_3_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_3(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_3_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_3.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  4
#define INPUT_SAMPLES   39
#define POOL_SIZE       8
#define POOL_STRIDE     8
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void max_pooling1d_3(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#else
#error "Unsupported activation function"
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_3_H_
#define _CONV1D_3_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      4
#define INPUT_SAMPLES       4
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef float conv1d_3_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_3(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_3_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_3.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      4
#define INPUT_SAMPLES       4
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 0
#define BIASES_SCALE_FACTOR 0
#define TMP_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void conv1d_3(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

#error "Data type unsupported by CMSIS-NN"

#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    4
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  3
#define CONV_GROUPS       1


const float  conv1d_3_bias[CONV_FILTERS] = {0x1.8f6c0e0000000p-3, 0x1.4b2b5e0000000p-1, -0x1.92efec0000000p-6, 0x1.adab620000000p-2, 0x1.7973e40000000p-4, -0x1.6ceec00000000p-1, -0x1.1d25ec0000000p-1, 0x1.0044da0000000p-1, 0x1.2455300000000p-1, -0x1.0140300000000p+0, -0x1.b40f580000000p-1, 0x1.1bd22c0000000p-3, 0x1.3846660000000p-1, 0x1.1d30680000000p-1, 0x1.41c4fa0000000p-3, -0x1.1931760000000p-2, -0x1.f3b4000000000p-4, -0x1.9309ba0000000p-4, 0x1.ef14920000000p-2, 0x1.1d2d420000000p-2, -0x1.ea78c20000000p-5, 0x1.a52afa0000000p-2, -0x1.2b09880000000p-2, -0x1.0c2b5a0000000p-3, -0x1.82d3a60000000p-4, 0x1.36b24c0000000p-1, -0x1.6acfd00000000p-5, 0x1.797caa0000000p-5, 0x1.b0dfe80000000p-3, -0x1.c341d20000000p-1, -0x1.67d3b60000000p-1, 0x1.89634a0000000p-4}
;

const float  conv1d_3_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-0x1.b508780000000p-3, 0x1.d8fe000000000p-4, 0x1.d22d420000000p-2, 0x1.f20aa00000000p-4}
, {0x1.800ee60000000p-4, 0x1.15cbaa0000000p-2, 0x1.dfe06e0000000p-5, -0x1.e17fcc0000000p-2}
, {0x1.1929f80000000p-2, 0x1.a940b40000000p-4, 0x1.26c2f60000000p-1, -0x1.e8b2740000000p-1}
}
, {{0x1.0c44880000000p-1, -0x1.11b0380000000p-3, 0x1.386a660000000p-2, 0x1.f702500000000p-4}
, {-0x1.624ddc0000000p-2, -0x1.1e2d2e0000000p-4, 0x1.d042fc0000000p-4, -0x1.5bb4300000000p-4}
, {0x1.5038660000000p-3, 0x1.d239dc0000000p-4, -0x1.0a33b00000000p-7, 0x1.2c85620000000p-4}
}
, {{0x1.3ccf240000000p-4, 0x1.103fec0000000p-2, -0x1.c876f40000000p-2, 0x1.935dd60000000p-2}
, {0x1.8df0260000000p-2, 0x1.0c4bbc0000000p-3, -0x1.6fb8b60000000p-2, 0x1.73e4b40000000p-3}
, {0x1.4037e00000000p-4, 0x1.2aaab60000000p-2, -0x1.b09cf20000000p-4, -0x1.13b1600000000p-3}
}
, {{-0x1.0cfcce0000000p-3, 0x1.6e9e920000000p-4, -0x1.d142140000000p-4, -0x1.8caaec0000000p-1}
, {-0x1.4d424e0000000p-4, 0x1.d8d9f40000000p-2, -0x1.00b2780000000p-4, -0x1.8c9fde0000000p-3}
, {0x1.0cd5f40000000p-2, 0x1.49278c0000000p-5, -0x1.31a7440000000p-4, -0x1.aa75880000000p-2}
}
, {{0x1.51462c0000000p-4, 0x1.6e7c2a0000000p-4, -0x1.8a8f0e0000000p-6, 0x1.eeb5f80000000p-4}
, {-0x1.2870040000000p-4, 0x1.48c3320000000p-2, -0x1.d0bb760000000p-4, 0x1.8880ca0000000p-6}
, {0x1.5b400c0000000p-2, 0x1.4cb5780000000p-2, -0x1.0219b40000000p-2, 0x1.a2fbf20000000p-3}
}
, {{-0x1.1a96c20000000p-1, -0x1.2df1c40000000p-3, -0x1.7fa1780000000p-2, 0x1.17ad000000000p-2}
, {-0x1.60548a0000000p-1, -0x1.1dd1960000000p-6, 0x1.6dc8820000000p-3, 0x1.1943440000000p-2}
, {-0x1.47ed340000000p-4, -0x1.7850400000000p-3, 0x1.5e9b960000000p-3, -0x1.a039c00000000p-3}
}
, {{-0x1.38e3f20000000p-2, 0x1.742a240000000p-2, -0x1.087ee60000000p-2, -0x1.2f64900000000p-2}
, {-0x1.3c387c0000000p-2, 0x1.9898800000000p-3, -0x1.309bfa0000000p-2, 0x1.5c36b40000000p-2}
, {-0x1.d715520000000p-1, 0x1.ac7e5c0000000p-4, -0x1.9a90160000000p-3, 0x1.f6d4580000000p-3}
}
, {{-0x1.08af620000000p-4, -0x1.77e9320000000p-8, -0x1.9564360000000p-3, 0x1.6aabca0000000p-6}
, {0x1.593f280000000p-3, 0x1.e718a20000000p-3, 0x1.478a8e0000000p-5, 0x1.5475ee0000000p-4}
, {0x1.41b0ec0000000p-4, -0x1.2676900000000p-4, 0x1.246ac40000000p-3, -0x1.772a540000000p-10}
}
, {{0x1.5223180000000p-3, 0x1.f78f4e0000000p-3, -0x1.45a95c0000000p-3, 0x1.99aa620000000p-4}
, {-0x1.8a6b660000000p-4, -0x1.7af6780000000p-3, 0x1.97c65e0000000p-3, 0x1.0bbb460000000p-3}
, {0x1.8ded4e0000000p-4, 0x1.08df640000000p-2, 0x1.78ac360000000p-2, -0x1.0988c60000000p-6}
}
, {{-0x1.16babe0000000p-1, -0x1.3554300000000p-4, 0x1.be4f0e0000000p-9, 0x1.2d3b920000000p-3}
, {-0x1.c522b40000000p-8, -0x1.ab6d960000000p-3, 0x1.a08cc00000000p-4, 0x1.074c4e0000000p-2}
, {-0x1.8ec5e40000000p-7, -0x1.dbae060000000p-3, -0x1.3a42200000000p-2, 0x1.c00a2a0000000p-2}
}
, {{-0x1.671c2e0000000p-2, -0x1.2b4de00000000p-3, -0x1.91f7d80000000p-3, 0x1.5877ee0000000p-2}
, {-0x1.79fe560000000p-2, -0x1.a9db0a0000000p-3, -0x1.7d5d480000000p-4, 0x1.5608f80000000p-2}
, {-0x1.55c19e0000000p-1, -0x1.7461ee0000000p-7, -0x1.b547640000000p-3, -0x1.50bfee0000000p-5}
}
, {{0x1.cb648c0000000p-3, 0x1.cf5c500000000p-3, 0x1.16d1a60000000p-2, -0x1.0946cc0000000p+0}
, {-0x1.cf61fe0000000p-5, 0x1.d648d20000000p-3, 0x1.576b040000000p-1, -0x1.08d03c0000000p-1}
, {-0x1.5660140000000p-3, -0x1.9509f60000000p-5, 0x1.99b4080000000p-4, -0x1.884e800000000p-1}
}
, {{0x1.70208a0000000p-3, 0x1.d13fb80000000p-3, 0x1.82c9b80000000p-3, 0x1.0d6d7c0000000p-3}
, {0x1.c3861a0000000p-2, -0x1.8396a40000000p-7, -0x1.0778680000000p-4, 0x1.4069200000000p-5}
, {0x1.e807940000000p-5, -0x1.2322f80000000p-2, -0x1.ee478e0000000p-3, -0x1.87fb940000000p-4}
}
, {{0x1.0abcaa0000000p-6, -0x1.19ce4c0000000p-2, 0x1.989e440000000p-2, -0x1.42fdec0000000p-7}
, {0x1.2649be0000000p-1, 0x1.44beea0000000p-3, 0x1.a0c8ac0000000p-2, 0x1.14ab680000000p-3}
, {-0x1.e3550c0000000p-5, -0x1.02f02e0000000p-2, -0x1.3ac00a0000000p-7, 0x1.eff17e0000000p-3}
}
, {{0x1.35786e0000000p-2, 0x1.628af40000000p-4, 0x1.155a820000000p-1, 0x1.e5667c0000000p-5}
, {0x1.70eb820000000p-2, 0x1.e39a1e0000000p-4, 0x1.7cb7680000000p-1, -0x1.80fad80000000p-3}
, {0x1.4bbdfc0000000p-3, -0x1.26767c0000000p-5, 0x1.35b3ea0000000p-1, -0x1.3346940000000p-3}
}
, {{-0x1.4655100000000p-2, 0x1.5a01bc0000000p-2, -0x1.65e54a0000000p-2, -0x1.499b060000000p-2}
, {-0x1.27ad840000000p-1, 0x1.ca57340000000p-3, -0x1.b114c60000000p-3, 0x1.09a9c40000000p-3}
, {-0x1.61901a0000000p-6, 0x1.5b298c0000000p-2, -0x1.aba55c0000000p-3, -0x1.71836e0000000p-2}
}
, {{0x1.1e13540000000p-3, 0x1.22b5ac0000000p-4, 0x1.b4b7640000000p-3, -0x1.d2db380000000p-3}
, {-0x1.6bed760000000p-2, 0x1.84a2c60000000p-7, -0x1.7a4c360000000p-3, 0x1.9022300000000p-5}
, {-0x1.46d5e80000000p-1, -0x1.1d71f60000000p-3, -0x1.b3c9a40000000p-3, 0x1.291f3a0000000p-2}
}
, {{-0x1.3d61a00000000p-2, 0x1.f245280000000p-3, -0x1.9c4c9c0000000p-5, -0x1.c8e4200000000p-4}
, {-0x1.5d5e800000000p-3, 0x1.15a07a0000000p-2, 0x1.fe7b260000000p-6, -0x1.4518e00000000p-2}
, {-0x1.6931000000000p-3, 0x1.da7ca20000000p-2, -0x1.1a00260000000p-5, -0x1.fa4ab60000000p-3}
}
, {{0x1.e355100000000p-2, -0x1.6358a00000000p-2, -0x1.a9908e0000000p-5, 0x1.5e080a0000000p-3}
, {-0x1.415e000000000p-2, -0x1.26b6e20000000p-2, 0x1.0eed300000000p-4, -0x1.5a68960000000p-5}
, {0x1.91b2c20000000p-3, 0x1.0bd1c60000000p-4, 0x1.e4d4400000000p-2, 0x1.41c8cc0000000p-2}
}
, {{-0x1.093b6e0000000p-2, -0x1.59cf280000000p-6, 0x1.1b30260000000p-3, 0x1.f528120000000p-3}
, {-0x1.d961620000000p-3, -0x1.924a120000000p-3, -0x1.9e44320000000p-4, 0x1.2793960000000p-1}
, {-0x1.486d0c0000000p-5, 0x1.074fca0000000p-8, 0x1.ece7b60000000p-4, 0x1.1714240000000p-4}
}
, {{0x1.33ddd80000000p-5, 0x1.55802c0000000p-5, -0x1.7d37820000000p-5, -0x1.1163920000000p-2}
, {-0x1.587d2a0000000p-3, -0x1.5e06b00000000p-3, 0x1.c868500000000p-4, 0x1.b3ee6a0000000p-5}
, {0x1.5467600000000p-3, -0x1.0d11f00000000p-5, -0x1.9588d40000000p-5, -0x1.a694420000000p-5}
}
, {{0x1.68e63a0000000p-1, 0x1.60c6b00000000p-4, 0x1.0bafda0000000p+0, -0x1.801f160000000p-3}
, {0x1.1317dc0000000p-2, -0x1.015a3a0000000p-2, 0x1.791b280000000p-2, -0x1.81682c0000000p-5}
, {0x1.2509280000000p-3, 0x1.9d5d600000000p-4, 0x1.833af80000000p-1, 0x1.1a1ff40000000p-3}
}
, {{-0x1.2c59540000000p-4, 0x1.9df2520000000p-2, -0x1.ffa3640000000p-8, -0x1.c1d3560000000p-2}
, {0x1.8c1e580000000p-6, 0x1.4f74380000000p-2, -0x1.f7672a0000000p-5, -0x1.5b62180000000p-2}
, {-0x1.f141860000000p-4, 0x1.e177000000000p-3, -0x1.36f8820000000p-3, 0x1.639ba60000000p-3}
}
, {{-0x1.01d3660000000p-4, 0x1.9272540000000p-6, -0x1.f13d900000000p-4, -0x1.6036ca0000000p-3}
, {0x1.4fb6de0000000p-3, -0x1.e4d41c0000000p-5, -0x1.a34c6a0000000p-9, -0x1.7dcb4a0000000p-2}
, {-0x1.3970d40000000p-3, -0x1.25cb380000000p-3, 0x1.136c660000000p-2, 0x1.f24e880000000p-3}
}
, {{-0x1.310fa20000000p-4, 0x1.238c200000000p-4, 0x1.0aeba60000000p-3, -0x1.7c3c900000000p-2}
, {-0x1.174a4a0000000p-5, -0x1.e17de60000000p-3, -0x1.27ae440000000p-3, 0x1.3578a60000000p-1}
, {0x1.358f480000000p-4, 0x1.4b389c0000000p-3, 0x1.55326a0000000p-3, -0x1.61fe880000000p-3}
}
, {{0x1.1ed0660000000p-2, -0x1.44aec40000000p-3, -0x1.5718320000000p-5, 0x1.39af380000000p-2}
, {0x1.4f8c660000000p-2, -0x1.1372120000000p-2, -0x1.15feec0000000p-3, 0x1.e938da0000000p-4}
, {-0x1.0dc4f60000000p-6, -0x1.c598b20000000p-3, 0x1.8cbd320000000p-2, 0x1.7dccd40000000p-5}
}
, {{-0x1.fd7ec20000000p-7, 0x1.4f6ae00000000p-3, -0x1.02d19e0000000p-1, 0x1.a3c1280000000p-5}
, {0x1.2a3a020000000p-7, 0x1.c859fa0000000p-3, -0x1.b71a1a0000000p-8, 0x1.7a77640000000p-3}
, {0x1.ab44300000000p-6, 0x1.41778c0000000p-6, -0x1.2469820000000p-6, -0x1.604cb00000000p-5}
}
, {{-0x1.6dd1440000000p-2, 0x1.8fe1800000000p-5, 0x1.87c6680000000p-6, 0x1.4510080000000p-2}
, {0x1.748d700000000p-6, -0x1.276a180000000p-3, -0x1.66554a0000000p-5, 0x1.e79caa0000000p-4}
, {-0x1.0820180000000p-3, -0x1.004c940000000p-4, 0x1.ebffca0000000p-3, 0x1.2a71a00000000p-2}
}
, {{-0x1.b027c20000000p-2, -0x1.70a81a0000000p-8, 0x1.0e297e0000000p-4, -0x1.0e56f00000000p-2}
, {-0x1.14ad180000000p-4, 0x1.4bbd7c0000000p-2, -0x1.133fc60000000p-5, -0x1.ee9a6c0000000p-4}
, {-0x1.62ebb20000000p-3, 0x1.d364260000000p-3, 0x1.24c0c20000000p-3, -0x1.977ff80000000p-3}
}
, {{-0x1.f570480000000p-4, -0x1.5ac5f00000000p-2, 0x1.202e120000000p-3, 0x1.34524e0000000p-3}
, {-0x1.acc92a0000000p-3, -0x1.2bc4ba0000000p-2, -0x1.4c13e80000000p-5, 0x1.542e2a0000000p-2}
, {0x1.22e6540000000p-3, -0x1.68c8f40000000p-3, -0x1.dde01c0000000p-3, 0x1.16a2c20000000p-2}
}
, {{-0x1.1e0d8a0000000p-1, 0x1.c1f1940000000p-4, -0x1.1215260000000p-3, 0x1.d3aefc0000000p-2}
, {-0x1.0e61980000000p-2, -0x1.74aab00000000p-6, -0x1.4ae5d20000000p-5, -0x1.6280bc0000000p-7}
, {-0x1.e6d7a00000000p-1, 0x1.19680e0000000p-3, -0x1.3918720000000p-2, -0x1.20fb8e0000000p-2}
}
, {{0x1.9524380000000p-3, 0x1.cbe5ec0000000p-3, 0x1.785ff00000000p-2, -0x1.b86f7c0000000p-1}
, {-0x1.84f7f00000000p-5, -0x1.bb02620000000p-5, -0x1.9dae740000000p-4, -0x1.f61e420000000p-3}
, {0x1.71a3800000000p-3, 0x1.1f31240000000p-4, 0x1.c6e03c0000000p-2, -0x1.1885000000000p-1}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_4_H_
#define _MAX_POOLING1D_4_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   2
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef float max_pooling1d_4_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_4(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_4_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_4.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   2
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void max_pooling1d_4(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#else
#error "Unsupported activation function"
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    flatten.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _FLATTEN_H_
#define _FLATTEN_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define OUTPUT_DIM 32

typedef float flatten_output_type[OUTPUT_DIM];

#if 0
void flatten(
  const number_t input[1][32], 			      // IN
	number_t output[OUTPUT_DIM]); 			                // OUT
#endif

#undef OUTPUT_DIM

#endif//_FLATTEN_H_
/**
  ******************************************************************************
  * @file    flatten.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 2.0.0
  * @date    26 november 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "flatten.h"
#include "number.h"
#endif

#define OUTPUT_DIM 32

#define NUMBER_T float
#define LONG_NUMBER_T float

static inline void flatten(
  const NUMBER_T input[1][32], 			      // IN
	NUMBER_T output[OUTPUT_DIM]) {			                // OUT

  NUMBER_T *input_flat = (NUMBER_T *)input;

  // Copy data from input to output only if input and output don't point to the same memory address already
  if (input_flat != output) {
    for (size_t i = 0; i < OUTPUT_DIM; i++) {
      output[i] = input_flat[i];
    }
  }
}

#undef OUTPUT_DIM
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    fc.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _DENSE_H_
#define _DENSE_H_

#ifndef SINGLE_FILE
#include "number.h"
#include <stdint.h>
#endif

#define INPUT_SAMPLES 32
#define FC_UNITS 3

typedef float dense_output_type[FC_UNITS];

#if 0
void dense(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]); 			                // OUT
#endif

#undef INPUT_SAMPLES
#undef FC_UNITS

#endif//_DENSE_H_
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "dense.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_SAMPLES 32
#define FC_UNITS 3
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 0
#define BIASES_SCALE_FACTOR 0
#define TMP_SCALE_FACTOR 0
#define INPUT_SCALE_FACTOR 0
#define OUTPUT_SCALE_FACTOR 0
#define OUTPUT_ROUND_MODE ROUND_MODE_NONE
#define NUMBER_T float
#define LONG_NUMBER_T float


static inline void dense(
  const NUMBER_T input[INPUT_SAMPLES], 			      // IN
	const NUMBER_T kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const NUMBER_T bias[FC_UNITS],			              // IN

	NUMBER_T output[FC_UNITS]) {			                // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short k, z; 
  LONG_NUMBER_T output_acc;

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0;
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ((LONG_NUMBER_T)kernel[k][z] * (LONG_NUMBER_T)input[z]);

    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
    // ReLU
    if (output_acc < 0) {
      output[k] = 0;
    } else {
#if defined(ACTIVATION_RELU6)
      if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
        output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
      }
#endif
      output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
#else
#error "Unsupported activation function"
#endif
  }
#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

#error "Data type unsupported by CMSIS-NN"

#endif
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 32
#define FC_UNITS 3


const float dense_bias[FC_UNITS] = {-0x1.ef05dc0000000p-3, -0x1.25f0800000000p-2, 0x1.3ae6580000000p-1}
;

const float dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{0x1.3f65ec0000000p-2, 0x1.aaea760000000p-3, -0x1.caefdc0000000p-2, -0x1.0ecf8e0000000p-4, -0x1.55e9bc0000000p-2, 0x1.11c9040000000p-1, 0x1.370d020000000p-5, -0x1.f87e6a0000000p-3, -0x1.92f3060000000p-5, 0x1.ef87260000000p-2, 0x1.ab03780000000p-1, 0x1.18e63e0000000p-2, -0x1.778e360000000p-2, 0x1.f344ec0000000p-6, 0x1.17ce6e0000000p-3, -0x1.042d720000000p-3, -0x1.19fc8a0000000p-6, 0x1.29687c0000000p-1, 0x1.7841360000000p-4, 0x1.4f1e740000000p-2, -0x1.7831b80000000p-2, 0x1.9564120000000p-2, 0x1.9b00740000000p-2, -0x1.a4b9ac0000000p-7, 0x1.bab5ca0000000p-2, -0x1.9362360000000p-2, -0x1.af452a0000000p-2, 0x1.bf9e180000000p-2, 0x1.9713480000000p-2, 0x1.96ea140000000p-1, 0x1.45c2ca0000000p-2, 0x1.246d740000000p-2}
, {-0x1.d0d1c20000000p-1, 0x1.2e54300000000p-4, 0x1.5160c40000000p-1, -0x1.48a67a0000000p-1, 0x1.1f0a720000000p-1, -0x1.3d34e80000000p-2, 0x1.11c84a0000000p-1, 0x1.e131860000000p-8, -0x1.7cf2340000000p-7, 0x1.3db3da0000000p-2, -0x1.c3865a0000000p-4, -0x1.458d3c0000000p-1, -0x1.56b07c0000000p-2, -0x1.89772e0000000p+0, -0x1.6d9cb40000000p+0, 0x1.8bd5540000000p-2, -0x1.c199400000000p-3, 0x1.38fad00000000p-1, -0x1.01c7200000000p+0, -0x1.cd41740000000p-1, 0x1.1e55080000000p-3, -0x1.4d9c800000000p+0, 0x1.0f4be60000000p-1, -0x1.275f4e0000000p-3, -0x1.62f9f20000000p-2, -0x1.c67c7c0000000p-1, 0x1.dfec2e0000000p-4, -0x1.100be20000000p-1, -0x1.2be43c0000000p-2, 0x1.7b89100000000p-3, 0x1.9a82ae0000000p-2, -0x1.2b8eb20000000p+0}
, {0x1.7306540000000p-3, 0x1.0c0ce00000000p-1, 0x1.4020ca0000000p-3, 0x1.79cb340000000p-1, 0x1.103daa0000000p-2, -0x1.7250560000000p-1, -0x1.26218e0000000p-1, 0x1.41477e0000000p-2, 0x1.7d6da00000000p-2, -0x1.109e9c0000000p+0, -0x1.1fd1c60000000p+0, 0x1.3211b80000000p-2, 0x1.2c960a0000000p-1, 0x1.fb8b1c0000000p-2, 0x1.5c23ce0000000p-2, -0x1.866e660000000p-1, -0x1.546ab20000000p-2, -0x1.9e87700000000p-1, 0x1.161f4e0000000p-1, -0x1.2c5a580000000p-3, -0x1.3e93380000000p-3, 0x1.0a0fe60000000p-1, -0x1.5866cc0000000p-1, 0x1.f20e4a0000000p-2, -0x1.29e44e0000000p-3, 0x1.8c07500000000p-1, -0x1.1dda5c0000000p-3, -0x1.8e9da20000000p-4, -0x1.e548ca0000000p-3, -0x1.03a1160000000p+0, -0x1.b4ddb60000000p+0, 0x1.1f2ec80000000p-4}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    model.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    08 july 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */


#ifdef __cplusplus
extern "C" {
#endif

#ifndef __MODEL_H__
#define __MODEL_H__

#ifndef SINGLE_FILE
#include "number.h"

 // InputLayer is excluded
#include "max_pooling1d.h" // InputLayer is excluded
#include "conv1d.h" // InputLayer is excluded
#include "max_pooling1d_1.h" // InputLayer is excluded
#include "conv1d_1.h" // InputLayer is excluded
#include "max_pooling1d_2.h" // InputLayer is excluded
#include "average_pooling1d.h" // InputLayer is excluded
#include "conv1d_2.h" // InputLayer is excluded
#include "max_pooling1d_3.h" // InputLayer is excluded
#include "conv1d_3.h" // InputLayer is excluded
#include "max_pooling1d_4.h" // InputLayer is excluded
#include "flatten.h" // InputLayer is excluded
#include "dense.h"
#endif


#define MODEL_INPUT_DIM_0 16000
#define MODEL_INPUT_DIM_1 1
#define MODEL_INPUT_DIMS 16000 * 1

#define MODEL_OUTPUT_SAMPLES 3

#define MODEL_INPUT_SCALE_FACTOR 0 // scale factor of InputLayer
#define MODEL_INPUT_ROUND_MODE ROUND_MODE_NONE
#define MODEL_INPUT_NUMBER_T float
#define MODEL_INPUT_LONG_NUMBER_T float

// node 0 is InputLayer so use its output shape as input shape of the model
// typedef  input_t[16000][1];
typedef float input_t[16000][1];
typedef dense_output_type output_t;


void cnn(
  const input_t input,
  output_t output);

void reset(void);

#endif//__MODEL_H__


#ifdef __cplusplus
} // extern "C"
#endif
/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"
// #include <chrono>

 // InputLayer is excluded
#include "max_pooling1d.c" // InputLayer is excluded
#include "conv1d.c"
#include "weights/conv1d.c" // InputLayer is excluded
#include "max_pooling1d_1.c" // InputLayer is excluded
#include "conv1d_1.c"
#include "weights/conv1d_1.c" // InputLayer is excluded
#include "max_pooling1d_2.c" // InputLayer is excluded
#include "average_pooling1d.c" // InputLayer is excluded
#include "conv1d_2.c"
#include "weights/conv1d_2.c" // InputLayer is excluded
#include "max_pooling1d_3.c" // InputLayer is excluded
#include "conv1d_3.c"
#include "weights/conv1d_3.c" // InputLayer is excluded
#include "max_pooling1d_4.c" // InputLayer is excluded
#include "flatten.c" // InputLayer is excluded
#include "dense.c"
#include "weights/dense.c"
#endif


void cnn(
  const input_t input,
  dense_output_type dense_output) {
  
  // Output array allocation
  static union {
    max_pooling1d_output_type max_pooling1d_output;
    max_pooling1d_1_output_type max_pooling1d_1_output;
    max_pooling1d_2_output_type max_pooling1d_2_output;
    conv1d_2_output_type conv1d_2_output;
    conv1d_3_output_type conv1d_3_output;
  } activations1;

  static union {
    conv1d_output_type conv1d_output;
    conv1d_1_output_type conv1d_1_output;
    average_pooling1d_output_type average_pooling1d_output;
    max_pooling1d_3_output_type max_pooling1d_3_output;
    max_pooling1d_4_output_type max_pooling1d_4_output;
    flatten_output_type flatten_output;
  } activations2;


// Model layers call chain 
  
  
  max_pooling1d( // First layer uses input passed as model parameter
    input,
    activations1.max_pooling1d_output
    );
  
  
  conv1d(
    activations1.max_pooling1d_output,
    conv1d_kernel,
    conv1d_bias,
    activations2.conv1d_output
    );
  
  
  max_pooling1d_1(
    activations2.conv1d_output,
    activations1.max_pooling1d_1_output
    );
  
  
  conv1d_1(
    activations1.max_pooling1d_1_output,
    conv1d_1_kernel,
    conv1d_1_bias,
    activations2.conv1d_1_output
    );
  
  
  max_pooling1d_2(
    activations2.conv1d_1_output,
    activations1.max_pooling1d_2_output
    );
  
  
  average_pooling1d(
    activations1.max_pooling1d_2_output,
    activations2.average_pooling1d_output
    );
  
  
  conv1d_2(
    activations2.average_pooling1d_output,
    conv1d_2_kernel,
    conv1d_2_bias,
    activations1.conv1d_2_output
    );
  
  
  max_pooling1d_3(
    activations1.conv1d_2_output,
    activations2.max_pooling1d_3_output
    );
  
  
  conv1d_3(
    activations2.max_pooling1d_3_output,
    conv1d_3_kernel,
    conv1d_3_bias,
    activations1.conv1d_3_output
    );
  
  
  max_pooling1d_4(
    activations1.conv1d_3_output,
    activations2.max_pooling1d_4_output
    );
  
  
  flatten(
    activations2.max_pooling1d_4_output,
    activations2.flatten_output
    );
  
  
  dense(
    activations2.flatten_output,
    dense_kernel,
    dense_bias,// Last layer uses output passed as model parameter
    dense_output
    );
}

#ifdef __cplusplus
} // extern "C"
#endif
