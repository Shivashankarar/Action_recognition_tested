/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include "gstnvdsmeta.h"
#include "nvds_analytics_meta.h"
#include "analytics.h"
#include "nvdspreprocess_meta.h"
#include "deepstream_action.h"

/** Defines the maximum size of a string. */
#define MAX_STR_LEN 2048

/** Defines the maximum size of an array for storing a text result. */
#define MAX_LABEL_SIZE 128

/** 3D model input NCDHW has 5 dims; 2D model input NSHW has 4 dims */
#define MODEL_3D_SHAPES 5

/* By default, OSD process-mode is set to GPU_MODE. To change mode, set as:
 * 0: CPU mode
 * 1: GPU mode
 */
#define OSD_PROCESS_MODE 1

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 1

/* Print FPS per every several frames*/
#define FPS_INTERVAL 30

/* Action recognition config */
static NvDsARConfig gActionConfig;
#define MAX_CLASS_LEN 7
static const gchar kActioClasseLabels[MAX_CLASS_LEN][MAX_LABEL_SIZE] = {
    "pull", "push" , "idle", "scoop", "pullback","loading","calibrating"};
/* custom_parse_nvdsanalytics_meta_data 
 * and extract nvanalytics metadata */
	extern "C" void
analytics_custom_parse_action_meta_data (NvDsBatchMeta *batch_meta)
{

  NvDsMetaList *l_user_meta = NULL;
  NvDsUserMeta *user_meta = NULL;
  for (l_user_meta = batch_meta->batch_user_meta_list; l_user_meta != NULL;
       l_user_meta = l_user_meta->next)
  {
    user_meta = (NvDsUserMeta *)(l_user_meta->data);
    if (user_meta->base_meta.meta_type == NVDS_PREPROCESS_BATCH_META)
    {
      GstNvDsPreProcessBatchMeta *preprocess_batchmeta =
          (GstNvDsPreProcessBatchMeta *)(user_meta->user_meta_data);
      std::string model_dims = "";
      if (preprocess_batchmeta->tensor_meta) {
        if (preprocess_batchmeta->tensor_meta->tensor_shape.size() == MODEL_3D_SHAPES) {
          model_dims = "3D: AR - ";
        } else {
          model_dims = "2D: AR - ";
        }
      }
      for (auto &roi_meta : preprocess_batchmeta->roi_vector)
      {
        NvDsMetaList *l_user = NULL;
        for (l_user = roi_meta.roi_user_meta_list; l_user != NULL;
             l_user = l_user->next)
        {
          NvDsUserMeta *user_meta = (NvDsUserMeta *)(l_user->data);
          if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META)
          {
            	NvDsInferTensorMeta *tensor_meta = (NvDsInferTensorMeta *)(user_meta->user_meta_data);
            	gfloat max_prob = 0;
            	gint class_id = -1;
            	gfloat *buffer = (gfloat *)tensor_meta->out_buf_ptrs_host[0];
            	for (size_t i = 0; i < tensor_meta->output_layers_info[0].inferDims.d[0]; i++)
            	{

              		if (buffer[i] > max_prob){
                		max_prob = buffer[i];
                		class_id = i;
              		}
            	}
            	const gchar *label = "";
            	if (class_id < MAX_CLASS_LEN)
              		label = kActioClasseLabels[class_id];
		NvDsObjectMeta *object_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
    		object_meta->object_id = roi_meta.frame_meta->source_id;
    		object_meta->confidence = max_prob,
    		g_strlcpy (object_meta->obj_label,label, MAX_LABEL_SIZE);
		object_meta->rect_params.left = roi_meta.roi.left;
		object_meta->rect_params.top = roi_meta.roi.top;
    		nvds_add_obj_meta_to_frame(roi_meta.frame_meta, object_meta, NULL);
    		roi_meta.frame_meta->bInferDone = TRUE;
	}
        }
	NvDsMetaList *l_classifier = NULL;
        for (l_classifier = roi_meta.classifier_meta_list; l_classifier != NULL;
             l_classifier = l_classifier->next)
        {
          NvDsClassifierMeta *classifier_meta = (NvDsClassifierMeta *)(l_classifier->data);
          NvDsLabelInfoList *l_label;
          for (l_label = classifier_meta->label_info_list; l_label != NULL;
               l_label = l_classifier->next)
          {
            NvDsLabelInfo *label_info = (NvDsLabelInfo *)l_label->data;

            NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
            display_meta->num_labels = 1;

            NvOSD_TextParams *txt_params = &display_meta->text_params[0];
            txt_params->display_text = (char *)g_malloc0(MAX_STR_LEN);

            snprintf(txt_params->display_text, MAX_STR_LEN - 1,"%s: %s", model_dims.c_str(), label_info->result_label);
            //LOG_DEBUG("classification result: cls_id: %d, label: %s", label_info->result_class_id, label_info->result_label);
            /* Now set the offsets where the string should appear */
            txt_params->x_offset = roi_meta.roi.left;
            txt_params->y_offset = (uint32_t)std::max<int32_t>(roi_meta.roi.top - 10, 0);

            /* Font , font-color and font-size */
            txt_params->font_params.font_name = (char *)"Serif";
            txt_params->font_params.font_size = 12;
            txt_params->font_params.font_color.red = 1.0;
            txt_params->font_params.font_color.green = 1.0;
            txt_params->font_params.font_color.blue = 1.0;
            txt_params->font_params.font_color.alpha = 1.0;

            /* Text background color */
            txt_params->set_bg_clr = 1;
            txt_params->text_bg_clr.red = 0.0;
            txt_params->text_bg_clr.green = 0.0;
            txt_params->text_bg_clr.blue = 0.0;
            txt_params->text_bg_clr.alpha = 1.0;

            nvds_add_display_meta_to_frame(roi_meta.frame_meta, display_meta);
          }
        }
      }
    }
}
}
