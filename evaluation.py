from itertools import product as it_product

import numpy as np
import pandas as pd
import tqdm
from torch.utils import data


class EvaluateDose:
    """Evaluate a full dose distribution against the reference dose on the OpenKBP competition metrics"""

    def __init__(self, data_src):
        """
        Prepare the class for evaluating dose distributions
        :param data_loader: a data loader object that loads data from the reference dataset
        :param dose_loader: a data loader object that loads a dose tensor from any dataset (e.g., predictions)
        """
        # Initialize objects
        self.data_src = data_src  # Loads data related to ground truth patient information

        # Initialize objects for later
        self.patient_id = None
        self.roi_mask = None
        self.new_dose = None
        #self.reference_dose = None
        self.voxel_size = None
        self.possible_dose_mask = None
        self.dose_score_vec = np.zeros(len(self.data_src))
        self.sample_idx = 0

        # Set metrics to be evaluated
        self.oar_eval_metrics = ['D_0.1_cc', 'mean']
        self.tar_eval_metrics = ['D_99', 'D_95', 'D_1']

        # Name metrics for data frame
        oar_metrics = list(it_product(self.oar_eval_metrics, self.data_src.rois['oars']))
        target_metrics = list(it_product(self.tar_eval_metrics, self.data_src.rois['targets']))

        # Make data frame to store dose metrics and the difference data frame
        self.metric_difference_df = pd.DataFrame(index=self.data_src.patient_id_list,
                                                 columns=[*oar_metrics, *target_metrics]) 
        self.dose_a_metric_df = self.metric_difference_df.copy()
        self.dose_b_metric_df = self.metric_difference_df.copy()

    def append_sample(self, dose_arr_a, batch, dose_arr_b=None):   
        
        # "dose_arr_a.shape = (batchsize, 1, 128, 128, 128)"
        batch_size = dose_arr_a.shape[0]

        for i in range(batch_size):
            
            # "self.roi_mask.shape = (10, 128, 128, 128)"
            self.roi_mask = batch['structure_masks'][i].numpy().astype(bool)

            # Save the patient list to keep track of the patient id
            self.patient_id = batch['patient_id'][i]

            # Get voxel size
            self.voxel_size = np.prod(batch['voxel_dimensions'][i].numpy())

            # Get the possible dose mask
            self.possible_dose_mask = batch['possible_dose_mask'][i].numpy()

            dose_a = dose_arr_a[i].flatten()
            self.dose_a_metric_df = self.calculate_metrics(self.dose_a_metric_df, dose_a)

            if dose_arr_b is not None:
                dose_b = dose_arr_b[i].flatten()
                self.dose_b_metric_df = self.calculate_metrics(self.dose_b_metric_df, dose_b)
                self.dose_score_vec[self.sample_idx] = np.sum(np.abs(dose_a - dose_b)) / np.sum(self.possible_dose_mask)
                self.sample_idx += 1

    def make_metrics(self):
        dvh_score = np.nanmean((np.abs(self.dose_a_metric_df - self.dose_b_metric_df).values))
        dose_score = self.dose_score_vec.mean()
        return dvh_score, dose_score

    def get_patient_dose_tensor(self, dose_batch):
        """Retrieves a flattened dose tensor from the input data_loader.
        :param data_loader: a data loader that load a dose distribution
        :return: a flattened dose tensor
        """
        # Load the dose for the request patient
        #dose_batch = data_loader.get_batch(patient_list=self.patient_list)
        dose_key = [key for key in dose_batch.keys() if 'dose' in key.lower()][0]  # The name of the dose
        dose_tensor = dose_batch[dose_key][0].numpy()  # Dose tensor
        dose_tensor = dose_tensor.flatten()
        return dose_tensor

    def get_constant_patient_features(self, rois_batch):
        """Gets the roi tensor
        :param idx: the index for the batch to be loaded
        """
        # Load the batch of roi mask
        #rois_batch = self.data_loader.get_batch(idx)
        self.roi_mask = rois_batch['structure_masks'][0].numpy().astype(bool)
        # Save the patient list to keep track of the patient id
        self.patient_list = rois_batch['patient_id']
        # Get voxel size
        self.voxel_size = np.prod(rois_batch['voxel_dimensions'].numpy())
        # Get the possible dose mask
        self.possible_dose_mask = rois_batch['possible_dose_mask'].numpy()


    def calculate_metrics(self, metric_df, dose):
        """
        Calculate the competition metrics
        :param metric_df: A DataFrame with columns indexed by the metric name and the structure name
        :param dose: the dose to be evaluated
        :return: the same metric_df that is input, but now with the metrics for the provided dose
        """
        # find which roi has values in 10 rois (max is 1 from 0,1)
        roi_exists = self.roi_mask.max(axis=(1, 2, 3))

        voxels_in_tenth_of_cc = np.maximum(1, np.round(100/self.voxel_size))  
        for roi_idx, roi in enumerate(self.data_src.full_roi_list):
            if roi_exists[roi_idx]:

                # roi_dose is get the values of those location in the roi_mask eg. brainstem
                """ roimask is  0-1 mask vector"""
                roi_mask = self.roi_mask[roi_idx, :, :, :].flatten() 
                """roi_dose is nonzero values vector eg. 442 out of 29072 values"""
                roi_dose = dose[roi_mask]
                """442""" 
                roi_size = len(roi_dose) 

                if roi in self.data_src.rois['oars']:
                    if 'D_0.1_cc' in self.oar_eval_metrics:
                        # Find the fractional volume in 0.1cc to evaluate percentile
                        fractional_volume_to_evaluate = 100 - voxels_in_tenth_of_cc/roi_size * 100
                        metric_eval = np.percentile(roi_dose, fractional_volume_to_evaluate)
                        metric_df.at[self.patient_id, ('D_0.1_cc', roi)] = metric_eval
                    if 'mean' in self.oar_eval_metrics:
                        metric_eval = roi_dose.mean()
                        metric_df.at[self.patient_id, ('mean', roi)] = metric_eval
                elif roi in self.data_src.rois['targets']:
                    if 'D_99' in self.tar_eval_metrics:
                        metric_eval = np.percentile(roi_dose, 1)
                        metric_df.at[self.patient_id, ('D_99', roi)] = metric_eval
                    if 'D_95' in self.tar_eval_metrics:
                        metric_eval = np.percentile(roi_dose, 5)
                        metric_df.at[self.patient_id, ('D_95', roi)] = metric_eval
                    if 'D_1' in self.tar_eval_metrics:
                        metric_eval = np.percentile(roi_dose, 99)
                        metric_df.at[self.patient_id, ('D_1', roi)] = metric_eval

        return metric_df
