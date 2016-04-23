#!/usr/bin/env python

"""Pipeline for generating submission."""

import ensemble_pipeline
import validate

import numpy as np
import os
import sys
import argparse


def main(output_file, model_name, systole_model, diastole_model):
    validate_data = os.path.join(ensemble_pipeline.PROCESSED_DATA_DIR, 'validate')
    print 'transform validate data'
    _, data, _ = ensemble_pipeline.transform_data(model_name, 'validate', validate_data, new_db=True)

    print 'predict systole'
    systole_cdfs = ensemble_pipeline.predict(systole_model, data)
    print 'pridict diastole'
    diastole_cdfs = ensemble_pipeline.predict(diastole_model, data)
    print 'writing submission'
    validate.write_submission(output_file, systole_cdfs, diastole_cdfs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file')
    parser.add_argument('model_name')
    parser.add_argument('systole_model')
    parser.add_argument('diastole_model')
    args = parser.parse_args()

    main(args.output_file, args.model_name, args.systole_model, args.diastole_model)
