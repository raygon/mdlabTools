import h5py
import pandas as pd
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from joblib import Parallel, delayed
import os
import time
import numpy as np
import tables
from sklearn import preprocessing
import fire

import mdlab.h5Tools as h5Tools
import mdlab.utilfx as utilfx

import pdb as ipdb
import pprint
ppr = pprint.PrettyPrinter()


class DataFrameHDF(object):
  """Data structure to maninpulate/store metadata as pandas DataFrames
  and data arrays as ndarrays in the same HDF5 file."""
  def __init__(self, fn, array_key='ndarray_data', dataframe_key='dataframe_data', mode='r', use_dask=True, build_references=True, use_mock_references=True):
    self.fn = fn
    self.dataframe = self._load_dataframe(self.fn, dataframe_key=dataframe_key)
    self._hdf_file, self.data = self._load_hdf_arrays(self.fn, mode=mode, array_key=array_key)
    if use_dask:
      array_data_dict = {}
      print('Parsing ndarrays into dask.arrays:')
      for k in h5Tools.h5_filtered_walk(self.data):
        k = k.replace(array_key, '').replace('/', '')
        print('\t', k)
        try:
          array_data_dict[k] = da.from_array(self.data[k], chunks=(100, *self.data[k].shape[1:]))
        except ValueError as e:
          array_data_dict[k] = da.from_array(self.data[k], chunks=(100,))
      self.data = array_data_dict
    self.use_dask = use_dask
    self.build_references = build_references
    self.use_mock_references = use_mock_references

    # Feb 28, 2019
    if self.build_references:
      if use_mock_references:
        self.rebuild_references()
      elif use_dask:
        self.rebuild_references()
      else:
        raise NotImplementedError('unsupported mode for build_references: must set `use_dask`=True or `use_mock_references`=True')

    # self.dataframe.iloc = _wrappedComputeIndexer(self.dataframe.iloc)
    # self.dataframe.loc = _wrappedComputeIndexer(self.dataframe.loc)

    self._check_representation_invariant()

  def __len__(self):
    return len(self.dataframe)

  def __getattr__(self, name):
    try:
      attr = getattr(self.dataframe, name)
    except Exception as e:
      try:
        attr = getattr(self.data, name)
      except Exception as e:
        raise e
    return attr

  def __getitem__(self, key):
    self._check_representation_invariant()
    data_source = self._get_data_source(key)
    return data_source[key]

  def __setitem__(self, key, value):
    try:
      self.dataframe[key] = value
    except Exception as e:
      self.data.create_dataset(key, data=value)
    self._check_representation_invariant()

  def __iter__(self):
    self._check_representation_invariant()
    return self._iter_helper().__iter__()

  def _iter_helper(self):
    return [*list(self.dataframe), *list(self.data)]

  def keys(self):
    self._check_representation_invariant()
    return list(self.__iter__())

  def values(self):
    self._check_representation_invariant()
    return [*self.dataframe.values(), *self.data.values()]

  def items(self):
    self._check_representation_invariant()
    return [*self.dataframe.items(), *self.data.items()]

  def has_key(self, key):
    self._check_representation_invariant()
    return key in self

  def rebuild_references(self):
    self.build_references = True
    self._populate_dataframe_with_ndarray_data_references()
    self._check_representation_invariant()

  def _populate_dataframe_with_ndarray_data_references(self):
    mock_str = 'mock ' if self.use_mock_references else ''

    print('Populating dataframes with %sreferences to on-disk datasets:' % mock_str)
    ### <populate dataframe subsets with references to on-disk datasets, if requested>
    for i, dset in enumerate(self.data):
      print('\t%slinking --> %s' % (mock_str, dset))
      if self.use_mock_references:
        dset_refs = self.dataframe.index.map(lambda x: mock_lookup(x, self.data[dset]))
      else:
        # I think the DataFrameHDF objects need to be loaded with dask, or otherwise have the lookup wrapped so it won't evaluate immediately
        dset_refs = self.dataframe.index.map(lambda x: self.data[dset][x])
      self.dataframe[dset] = dset_refs
    ### </populate dataframe subsets with references to on-disk datasets, if requested>

  def _check_representation_invariant(self):
    out = True
    out = self._verify_unique_keys() and out

    if not out:
      raise ValueError('_check_representation_invariant is violated ')

    return out

  def _verify_unique_keys(self):
    # keys must be unique
    out = True
    if not self.build_references:
      if len(set(self._iter_helper())) != len(self._iter_helper()):
        out = False
        raise KeyError('keys must be unique if not building_references:')
    return out

  def _get_data_source(self, key):
    self._check_representation_invariant()

    if key in self.dataframe:
      out = self.dataframe
    elif key in self.data:
      out = self.data
    else:
      raise KeyError('key not found: %s' % key)

    return out

  def _load_dataframe(self, fn, **kwargs):
    return pd.read_hdf(fn, **kwargs)

  def _load_hdf_arrays(self, fn, mode='r', array_key=None, n_attempts=60):
    for i in range(n_attempts):
      try:
        h5_file = h5py.File(fn, mode=mode)
        h5_group = h5_file['/'] if array_key is None else h5_file[array_key]
        return h5_file, h5_group
      except Exception as e:
        print('DataFrameHDF HDF load failure %s/%s: %s' % (i + 1, n_attempts, e))
        time.sleep(np.random.rand())

  def masked_lookup(self, key, mask=None, dask_compute=False):
    if isinstance(mask, pd.DataFrame):
      _mask = mask.index
      # return self[key][_mask] # breaks with non-decreasing HDF index
      # return pd.Series({i: self[key][self.index.get_loc(i)] for i in _mask})  #TODO: reset index before writing source BigDataFrame !!! This probably doesn't work on if dataframe is a sample or head!!!
      if dask_compute:
        return pd.Series({i: self[key][i].compute() for i in _mask}) #TODO: does dask handle this automatically?
      else:
        return pd.Series({i: self[key][i] for i in _mask}) #TODO: does dask handle this automatically?
    else:
      return self[key][mask]

  def close(self):
    return self._hdf_file.close()

  def to_pdh5(self, wfn, array_mask=None, reset_index=True):
    if self.build_references:
      return self._to_dataframehdf_from_built_references(self, wfn)
    else:
      return self.to_dataframehdf_HACK(wfn, self.data, self.dataframe, array_mask=array_mask, reset_index=reset_index)

  def to_pdhdf(self, wfn, array_mask=None, reset_index=True):
    if self.build_references:
      self._to_dataframehdf_from_built_references(self, wfn)
    else:
      return self.to_dataframehdf_HACK(wfn, self.data, self.dataframe, array_mask=array_mask, reset_index=reset_index)

  @staticmethod
  def _to_dataframehdf_from_built_references(pdf, wfn, reset_index=True, n_jobs=3, parallel_backend='joblib'):
    if not isinstance(pdf, DataFrameHDF) and not isinstance(pdf, pd.DataFrame):
      ipdb.set_trace()
      raise ValueError('pdf input must be a DataFrameHDF or DataFrame')

    ndarray_columns = [x for x in pdf.data]

    # ### <write /ndarray_data>
    # print('writing pdf to hdf5: /ndarray_data:')
    # with tables.open_file(wfn, mode='w') as h5wf:
    #   for dset in ndarray_columns:
    #     print("\twriting '/%s':" % dset)
    #     temp_data = pdf[dset].iloc[0]()  #HACK
    #     data_storage = h5wf.create_vlarray('/ndarray_data', dset, tables.Atom.from_dtype(temp_data[0].dtype), createparents=True)

    #     for i, (row_ind, row) in enumerate(pdf.iterrows()):
    #       utilfx.print_replace('\t\t%s/%s' % (i + 1, len(pdf)))
    #       arr_val = row[dset]()
    #       data_storage.append(arr_val)
    # ### <write /ndarray_data>

    # ### <write /dataframe_data>
    # print('\nwriting pdf to hdf5: /dataframe_data:')
    # df_only = pdf.drop(columns=ndarray_columns)
    # print('\tNot writing columns: ', set(pdf.columns) - set(df_only.columns))
    # if reset_index:
    #   df_only['_pre_to_pdh5_index'] = df_only.index
    #   df_only.reset_index(inplace=True, drop=True)
    # utilfx.wrapped_write(df_only.to_hdf)(wfn, '/dataframe_data', format='f',  mode='a')
    # ### </write /dataframe_data>

    def _write_pdh5_chunk(df, wfn, data_rfn=None, data_columns=None, recipe_limit_to_slice=None, reset_index=True, wfn_stem=None):
    # def _write_pdh5_chunk(pdf_subset, wfn, wfn_stem=None):
      wfn_stem = '' if wfn_stem is None else wfn_stem
      wfn += '.SUBSET_%s' % wfn_stem

      # print(data_rfn)
      pdf_subset = DataFrameHDF(data_rfn, use_dask=False, build_references=False, use_mock_references=True)
      pdf_subset.dataframe = df
      # data_store = h5py.File(data_rfn, 'r')
      if data_columns is not None:
        raise NotImplementedError()
      pdf_subset.rebuild_references()

      with tables.open_file(wfn, mode='w') as h5wf:
        print('writing pdf_subset to hdf5: /ndarray_data:')
        for dset in ndarray_columns:
          print("\twriting '/%s':" % dset)
          temp_data = pdf_subset[dset].iloc[0]()  #HACK
          data_storage = h5wf.create_vlarray('/ndarray_data', dset, tables.Atom.from_dtype(temp_data[0].dtype), createparents=True)

          for i, (row_ind, row) in enumerate(pdf_subset.iterrows()):
            utilfx.print_replace('\t\t%s/%s' % (i + 1, len(pdf_subset)))
            arr_val = row[dset]()
            data_storage.append(arr_val)

        print('\nwriting pdf_subset to hdf5: /dataframe_data:')
        df_only = pdf_subset.drop(columns=ndarray_columns)
        print('\tNot writing columns: ', set(pdf_subset.columns) - set(df_only.columns))
        # reset_index = True  # this should always be set to True in this case # not in parallel context
        if reset_index:
          df_only['_pre_split_index'] = df_only.index
          df_only.reset_index(inplace=True, drop=True)
        utilfx.wrapped_write(df_only.to_hdf)(wfn, '/dataframe_data', format='f',  mode='a')

      pdf_subset.close()

    # grouped_inds = list(utilfx.grouper(pdf.index, n_groups=n_jobs))
    # print(len(grouped_inds))
    # ipdb.set_trace()
    # Parallel(n_jobs=n_jobs)(delayed(_write_pdh5_chunk)(pdf.loc[loc_inds], wfn, wfn_stem='%s_' % i) for i, loc_inds in enumerate(grouped_inds))

    chunk_size = int(np.ceil(pdf.shape[0] / n_jobs))
    num_remaining = pdf.shape[0]
    chunk_start_ind, chunk_end_ind = 0, 0
    slice_inds_list = []
    while num_remaining > 0:
      print('n_stim to generate: %s' % num_remaining, flush=True)
      temp_chunk_size = num_remaining if num_remaining < chunk_size else chunk_size
      chunk_end_ind += temp_chunk_size
      slice_inds_list.append([chunk_start_ind, chunk_end_ind])
      chunk_start_ind = chunk_end_ind
      num_remaining -= temp_chunk_size

    df_for_worker = pdf.dataframe.drop(columns=[x for x in pdf.data])
    ipdb.set_trace()
    Parallel(n_jobs=n_jobs)(delayed(_write_pdh5_chunk)(df_for_worker.iloc[slice(*loc_inds)], wfn, data_rfn=pdf._hdf_file.filename, data_columns=None, reset_index=True, wfn_stem='%s' % i) for i, loc_inds in enumerate(slice_inds_list))
    return wfn

  @staticmethod
  def to_dataframehdf_HACK(wfn, array_data, dataframe_data, array_mask=None, reset_index=True):
    # write hdf5 array data
    print('HACK writing ndarray data to hdf5')
    if array_data is None:
      with h5py.File(wfn, 'w') as h5wf:
        h5wf.create_group('/ndarray_data')
    else:
      try:
        with ProgressBar():
          if array_mask is None:
            utilfx.wrapped_write(da.to_hdf5)(wfn, {os.path.join('/ndarray_data', k): array_data[k] for k in array_data})
          else:
            utilfx.wrapped_write(da.to_hdf5)(wfn, {os.path.join('/ndarray_data', k): array_data[k][array_mask] for k in array_data})
      except Exception as e:
        raise e

    # write dataframe data
    print('\nwriting dataframe data to hdf5')
    if reset_index:
      dataframe_data['_pre_split_index'] = dataframe_data.index
      dataframe_data.reset_index(inplace=True, drop=True)
    utilfx.wrapped_write(dataframe_data.to_hdf)(wfn, '/dataframe_data', format='f',  mode='a')


def mock_lookup(key, data_store=None):
  def fx(dict_like=None):
    dict_like = data_store if dict_like is None else dict_like
    out = dict_like[key]
    if isinstance(out, da.core.Array):
      out = out.compute()
    return out
  return fx


def encode_labels_to_ints(df, keys_to_encode_to_int, append=True, verbose=1):
  label_encoders = {}
  for k in keys_to_encode_to_int:
    if k not in df.columns:
      continue
    utilfx.printiv('encoding column "%s" to int' % k, verbose)
    k_int = k + '_int'
    le = preprocessing.LabelEncoder()
    le.fit(sorted(df[k].values))
    if append:
      df[k_int] = le.transform(df[k].values)
    label_encoders[k] = le
  return label_encoders


def append_null_class_to_df(df_fn, columns=['corpus', 'path', 'source'], null_class_str='__nullClass__', wfn=True, append_ndarray_data=True, df_null_row=None, **kwargs):
  AUDIOSET_HACK = kwargs.pop('audioset_hack', False)

  if df_fn.endswith('h5') or df_fn.endswith('hdf5'):
    df = pd.read_hdf(df_fn)
  else:
    df = pd.read_pickle(df_fn)

  if wfn is True:
    wfn = df_fn

  columns = utilfx.make_iterable(columns)

  if null_class_str not in np.unique(df[columns[0]]) or df_null_row is not None:
    print('Appending nullClass: %s to dataframe: %s' % (null_class_str, df_fn))
    if df_null_row is None:
      null_class_columns = {k: null_class_str for k in columns}

      df_null_row = pd.DataFrame({
                                  **null_class_columns,
                                 'sr': df.iloc[0]['sr'],  #TODO: fix this hack
                                 }, index=[df.iloc[-1].name+1])

      if AUDIOSET_HACK:
        null_class_columns.update({})
    else:
      df_null_row = pd.DataFrame(df_null_row.values, index=[df.iloc[-1].name+1])

    df = df.append(df_null_row, ignore_index=False, sort=True)

    if wfn:
      print('writing nullClass metadata')
      # write updated dataframe metdata
      utilfx.wrapped_write(df.to_hdf)(wfn, '/dataframe_data', mode='a')

      if append_ndarray_data:
        with h5py.File(df_fn, 'a') as hd:
          hd_data = hd['/ndarray_data']
          for k in hd_data:
            gn = os.path.join('/ndarray_data', k)
            dset = hd_data[gn]

            dset.resize(dset.shape[0]+1, axis=0)  # 0 is the batch dimension
            # dset[-1, ...] = np.zeros(dset[-1, ...].shape)

  return df


def balance_dataframe(df, column, group_weights={}, verbose=1):
  # if isinstance(df, DataFrameHDF):
  #   bdf = df
  #   df = df.dataframe

  weights_column = 'weights_%s' % column
  df[weights_column] = np.nan

  # df_grouped = df.groupby(column)
  # per_class_weight = 1 / len(df_grouped)

  df_grouped = df.groupby(column)
  if group_weights is None:  # equal weighting between class groups
    df_test = df.get(['path', *column])  # use 'path' for counts column
    per_class_weight = 1 / len(df_grouped)
    df_test['group_weight'] = per_class_weight
  elif isinstance(group_weights, str):  # use group_weights column from df dataframe
    df_test = df.get(['path', *column, group_weights])  # use 'path' for counts column
    df_test.rename(column={group_weights: 'group_weight'}, inplace=True)
  else:  # should be dict of {group name(s): weight}; remaining probability mass will be split equally between unspecified groups
    # total_weight = np.sum([x for x in group_weights.values()])
    total_weight = np.sum([x[-1] for x in group_weights])
    if total_weight > 1:
      raise ValueError('group_weights sum cannot be larger than 1')
    remaining_weight = 1 - total_weight
    print(total_weight, remaining_weight)
    df_test = df.get(['path', *column])
    df_test['group_weight'] = remaining_weight / (len(df_grouped) - len(group_weights))

    # group_weights = [('__null__', '__null__')]
    # # query_dict = [list(toolz.interleave([column, x])) for x in group_weights]
    # group_weights_str = [list(map(lambda xx: '"%s"' % xx, x)) for x in group_weights]
    # print(group_weights_str)
    # # query_dict = [urlencode(dict(zip(column, x))) for x in group_weights_str]
    # query_str = [dict(zip(column, x)) for x in group_weights_str]
    # query_str = [urlencode(x).replace('%22', '"').replace('=', '==') for x in query_str]
    # # print(query_dict)
    # # query_str = urlencode(query_dict)
    print(group_weights)
    # query_str_list = {make_pandas_query_str(column, v[:-1]): v[-1] for v in group_weights}
    query_str_list = [(make_pandas_query_str(column, v[:-1]), v[-1]) for v in group_weights]
    # df_test['group_weight'] = remaining_weight / (len(df_grouped) - len(df_test.query()))
    print(query_str_list)
    for k, v in query_str_list:
      df_test['group_weight'].loc[df_test.query(k).index] = v
    # df_test['group_weight'].loc[df_test.query(query_str_list[0]).index] = 0.04
    # df_test['group_weight'].loc[df_test.query('speaker == "__null__" & word == "__null__"').index] = 0.04
    # display(df_test.tail())
    # ipdb.set_trace()
  df_test.rename(columns={'path': 'group_count'}, inplace=True)

  df_grouped = df_test.groupby(column)

  # if group_weights is None:  # equal weighting between class groups
  #   per_class_weight = 1 / len(df_grouped)
  #   df_grouped['group_weight'] = per_class_weight
  #   # class_weights = {x: 1 / len(df[column].unique()) for x in df[column].unique()}  # balanced corpora
  # elif isinstance(group_weights, str):  # use group_weights column from df dataframe
  #   df_grouped['group_weight'] = df[group_weights]
  # else:
  #   df_grouped['group_weight'] = corpus_weights

  group_counts = df_grouped['group_count'].count()
  df = df.join(group_counts, on=column)  # shouldn't require suffixes
  # df[weights_column] = per_class_weight / df['group_count']
  df[weights_column] = df_test['group_weight'] / df['group_count']
  # display(df)

  # for col, df_subset in df_grouped:
  #   print(col)
  #   print(len(df_subset))
  #   subset_weight = per_class_weight / len(df_subset)  # equal weighting between classes
  #   # ipdb.set_trace()
  #   df[weights_column].loc[df_subset.index] = subset_weight
  #   # df[weights_column][mask_col] = class_weights[col] / mask_col.sum()  # equal weighting between classes
  #   break

  if verbose > 0:
    print('`weights` column sum:', df[weights_column].sum())

  if verbose > 0:
    print('post weighted dataframe summary --> balance_column: %s, weights_column: %s' % (column, weights_column))
    # print(df[[weights_column, column]].groupby(column).sum()) # pre May 11, 2018
    # print(df.dataframe[[weights_column, column]].groupby(column).sum())  # make compatible with DataFrameHDF

  # if isinstance(df, DataFrameHDF):
  #   bdf.dataframe = df
  #   df = bdf

  return df, weights_column


def add_weights_to_dataframe(df, column, corpus_weights=None, verbose=1):
  """Weight each sample (row) in the dataframe, balanced based on the
  groups in `column`.

  Args:
    df (pd.DataFrame): Source dataframe to add a weights column to; will
      be operated in place.
    column (str): The name of the column to balance.
    corpus_weights (None, {group_name (str): group_weight (float)}, optional):
      Determine how to weight the groups to be balanced. If None, groups
      will be balanced equally; otherwise, use a dict of group_name:group_weight
      pairs to define the group class weights.
    verbose (int, optional): Print extra output, if requested.

  Returns:
    pd.DataFrame: updated dataframe with weights column
    str: the name of the created weights column
  """
  # if isinstance(df, DataFrameHDF):
  #   bdf = df
  #   df = df.dataframe

  weights_column = 'weights_%s' % column
  df[weights_column] = np.nan
  raw_weights = df.groupby(column).count()['path'] / len(df)

  if corpus_weights is None:
    class_weights = {x: 1 / len(df[column].unique()) for x in df[column].unique()}  # balanced corpora
  else:
    class_weights = corpus_weights

  if verbose > 0:
    print('Corpus Sampling Weights -->', class_weights)

  for col, w in raw_weights.iteritems():
    mask_col = df[column] == col
    df[weights_column][mask_col] = class_weights[col] / mask_col.sum()  # equal weighting between classes

  if verbose > 0:
    print('`weights` column sum:', df[weights_column].sum())

  if verbose > 0:
    print('post weighted dataframe summary --> balance_column: %s, weights_column: %s' % (column, weights_column))
    # print(df[[weights_column, column]].groupby(column).sum()) # pre May 11, 2018
    # print(df.dataframe[[weights_column, column]].groupby(column).sum())  # make compatible with DataFrameHDF

  # if isinstance(df, DataFrameHDF):
  #   bdf.dataframe = df
  #   df = bdf

  return df, weights_column


def convert_dataframe_pickle_to_dataframehdf(df, ndarray_columns=None, wfn=None, resample=None, guess_ndarray_names=True, hdf_backend_mode='vlarray'):
  if not isinstance(df, pd.DataFrame):
    df_rfn = df
    df = pd.read_pickle(df_rfn)
    wfn = wfn if wfn is not None else os.path.splitext(df_rfn)[0] + '.pdh5'

  if wfn is None:
    raise ValueError('`wfn` cannot be None when `df` is a DataFrame.')
  utilfx.touch_dir(os.path.dirname(wfn))

  ### <try to guess the ndarray columns in the dataframe>
  if ndarray_columns is None and guess_ndarray_names:
    ndarray_columns = [col for col in df.columns if isinstance(df[col].iloc[0], np.ndarray)]
  ### </try to guess the ndarray columns in the dataframe>

  ### </extract and write the array data to an hdf5 array>
  ndarray_data_storage = {}
  with tables.open_file(wfn, mode='w') as h5wf:
    for i, (row_ind, row) in enumerate(df.iterrows()):
      # print(i, row)
      utilfx.print_replace('%s/%s' % (i + 1, len(df)))

      # sr = row['sr']
      # s = row['signal']
      # print(s)

      # if s.ndim != 1:
      #   raise ValueError('expected a single data channel, found %s' % s.ndim)

      # if resample is not None and sr != resample:
      #   raise NotImplementedError()
      #   s = scipy.signal.resample_poly(s, resample, sr)
      #   sr = resample
      #   row['sr'] = sr
      #   row['signal'] = s

      # if i == 0:
        # ndarray_data_storage = {k: h5wf.create_vlarray('/ndarray_data', k, tables.Atom.from_dtype(df[k].dtype), createparents=True) for k in ndarray_columns}
        # data_storage = h5wf.create_vlarray('/ndarray_data', 'signal', tables.Atom.from_dtype(s.dtype), createparents=True)
#         data_storage = hdwf.create_carray('/ndarray_data', 'signal',
#                                           tables.Atom.from_dtype(s.dtype),
#                                           shape=[len(df), len(s)],
#   #                                             filters=filters,
#                                           createparents=True
#                                          )
      for k in ndarray_columns:
        temp_data = row[k]

        try:
          temp_data.shape
        except AttributeError as e:
          temp_data = temp_data[0]  # maybe the array is in a list?

        if temp_data.ndim != 1:
          raise ValueError('expected a single data channel, found %s' % temp_data.ndim)

        if i == 0:
          if hdf_backend_mode == 'vlarray':
            ndarray_data_storage[k] = h5wf.create_vlarray('/ndarray_data', k, tables.Atom.from_dtype(temp_data.dtype), createparents=True)
          elif hdf_backend_mode == 'carray':
            ndarray_data_storage[k] = h5wf.create_carray('/ndarray_data', k,
                                            tables.Atom.from_dtype(temp_data.dtype),
                                            shape=[len(df), len(temp_data)],
                                            createparents=True
                                           )
          else:
            raise ValueError('unrecognized hdf_backend_mode: %s' % hdf_backend_mode)
        if hdf_backend_mode == 'vlarray':
          ndarray_data_storage[k].append(temp_data)
        elif hdf_backend_mode == 'carray':
          ndarray_data_storage[k][i, :] = temp_data
        else:
          raise ValueError('unrecognized hdf_backend_mode: %s' % hdf_backend_mode)
#       break
  ### </extract and write the array data to an hdf5 array>

  ### <write updated dataframe to the same file>
  df_new = df.drop(columns=ndarray_columns)
  utilfx.wrapped_write(df_new.to_hdf)(wfn, '/dataframe_data', mode='a')
  ### </write updated dataframe to the same file>


def stack_dataframehdf(fntp, wfn_stem=None):  #dir_or_fntp
  if os.path.isdir(fntp):
    in_dir = fntp
    fntp = [os.path.join(fntp, f) for f in os.listdir(fntp) if f.endswith('.pdh5')]

    if not wfn_stem:
      wfn_stem = os.path.join(in_dir, 'stackedDataframeHDF')
  fntp = sorted(fntp)

  wfn = '%s_n%s_%s.pdh5' % (wfn_stem, len(fntp), utilfx.hash_str(str(fntp)))

  all_df = None
  h5wf = None
  ndarray_data_storage = {}
  with tables.open_file(wfn, mode='w') as h5wf:
    for i, rfn in enumerate(fntp):
      utilfx.print_replace('%s/%s' % (i, len(fntp)))
      df_temp = pd.read_hdf(rfn)

      if i == 0:
        all_df = df_temp
        with h5py.File(rfn, 'r') as h5rf:
          for k in h5rf['/ndarray_data']:
            temp_data = h5rf['/ndarray_data'][k][0]
            ndarray_data_storage[k] = h5wf.create_vlarray('/ndarray_data', k, tables.Atom.from_dtype(temp_data.dtype), createparents=True)

      # ppr.pprint(ndarray_data_storage)

      with h5py.File(rfn, 'r') as h5rf:
        for k in h5rf['/ndarray_data']:
          temp_data = h5rf[os.path.join('/ndarray_data', k)].value
          for a_ind in range(len(temp_data)):
            ndarray_data_storage[k].append(temp_data[a_ind])

      if i != 0:
        all_df = pd.concat((all_df, df_temp))

  wfn, _ = utilfx.wrapped_write(all_df.to_hdf)(wfn, '/dataframe_data', mode='a')
  return wfn


def write_subset_dataframehdf(pdf_subset, wfn, reset_index=True, ndarray_data_hdf=None, pre_write_fx=None):
  try:
    h5rf = pdf_subset.data
  except Exception:
    #TODO: cleanup h5 if loaded
    print('write_subset_dataframehdf: using provided ndarray_data_hdf...')
    if isinstance(ndarray_data_hdf, str):
      h5rf = h5py.File(ndarray_data_hdf, 'r')['/ndarray_data']
      close_hdf = True
    else:
      h5rf = ndarray_data_hdf
      close_hdf = False

  with tables.open_file(wfn, mode='w') as h5wf:
    for k in h5rf['/ndarray_data']:
      print("writing '/%s':" % k)
      temp_data = h5rf['/ndarray_data'][k]
      data_storage = h5wf.create_vlarray('/ndarray_data', k, tables.Atom.from_dtype(temp_data[0].dtype), createparents=True)

      for i, (loc_ind, df_row) in enumerate(pdf_subset.iterrows()):
        utilfx.print_replace('\t%s/%s' % (i+1, len(pdf_subset)))
        out_arr = pre_write_fx(temp_data[loc_ind]) if callable(pre_write_fx) else temp_data[loc_ind]
        data_storage.append(out_arr)

  if reset_index:
    pdf_subset['_pre_subset_index'] = pdf_subset.index
    pdf_subset = pdf_subset.reset_index(drop=True)
  wfn, _ = utilfx.wrapped_write(pdf_subset.to_hdf)(wfn, '/dataframe_data', mode='a')

  if close_hdf:
    h5rf.file.close()

  return wfn


def map_dataframehdf(pdf_fn, fx, wfn=None, columns='all', reset_index=True, pre_fx=None, chunksize=1, update_row=True):
  if chunksize != 1 and pre_fx:
    raise ValueError('pre_fx cannot be used when chunksize is not 1')

  if wfn is True:
    wfn = os.path.join(os.path.dirname(pdf_fn), 'map_dataframehdf', '%s-%s' % (fx.__name__, os.path.basename(pdf_fn)))
  if not wfn:
    raise ValueError('wfn arg must be provided; use wfn=True to attempt auto-naming')

  pdf = DataFrameHDF(pdf_fn, use_dask=False)
  df = pdf.dataframe
  hd = pdf.data

  if columns.lower() == 'all':
    columns = list(hd)
  else:
    columns = utilfx.make_iterable(columns)

  # constrain by dataframe metadata
  df = pre_fx(df) if callable(pre_fx) else df

  # map fx over constrained signals
  with tables.open_file(wfn, mode='w') as h5wf:
    for k in columns:  # apply fx to ever desired column
      print("writing '/%s':" % k)
      temp_data = hd[k]
      # data_storage = h5wf.create_vlarray('/ndarray_data', k, tables.Atom.from_dtype(temp_data[0].dtype), createparents=True) #Jan 10, 2019: this might cause issues with int16 and resampling
      data_storage = h5wf.create_vlarray('/ndarray_data', k, tables.Atom.from_dtype(np.arange(1, dtype=float).dtype), createparents=True)

      if chunksize == 1:
        for iloc_ind, (loc_ind, df_row) in enumerate(df.iterrows()):
          # if loc_ind == 10:
          #   break
          utilfx.print_replace('\t%s/%s' % (iloc_ind+1, len(df)))
          if callable(fx):
            out_arr, out_row = fx(temp_data[loc_ind], df_row)
            # update df_row metadata
            if update_row:
              # df.loc[loc_ind][:] = out_row  # this seems faster than below, but is still slow, TODO: address this Jan 9, 2019: this doesn't seem to update the dataframe now
              df.loc[loc_ind] = out_row  # this is a heavy op if the dataframe is big
          else:
            out_arr = temp_data[loc_ind]
          data_storage.append(out_arr)
      else:
        start_ind = 0
        end_ind = 0
        chunk_ctr = 0
        while end_ind < len(df):
          print('chunk: %s/%s' % (end_ind, len(df)//chunksize))
          chunk_ctr += 1
          start_ind = end_ind
          end_ind += chunksize
          end_ind = min(end_ind, len(df) - 1)  # limit to valid indices

          df_chunk = df.iloc[start_ind:end_ind]
          data_chunk = temp_data[start_ind:end_ind]
          print(data_chunk.dtype, len(data_chunk))

          for i, (loc_ind, df_row) in enumerate(df_chunk.iterrows()):
            # utilfx.print_replace('\t%s/%s' % (i+1, len(df_chunk)))
            if callable(fx):
              # ipdb.set_trace()
              out_arr, out_row = fx(data_chunk[i], df_row)
              if update_row:
                # df.loc[loc_ind][:] = out_row
                df.loc[loc_ind] = out_row
            else:
              raise ValueError()
              out_arr = data_chunk[i]
            data_storage.append(out_arr)

  # ipdb.set_trace()
  if reset_index:
    pdf['_pre_map_index'] = pdf.index
    pdf = pdf.reset_index(drop=True)
  # wfn, _ = utilfx.wrapped_write(pdf.to_hdf)(wfn, '/dataframe_data', mode='a') #TODO: figure this out
  wfn, _ = utilfx.wrapped_write(df.to_hdf)(wfn, '/dataframe_data', mode='a')

  return wfn


#################################
###---------- TESTS ----------###
#################################
def run_tests():
  rfn = '/om/user/raygon/projects/public/jsinDataset/assets/data/processed/v2Balanced/sr_20000/splits/train_stackedDataframeHDF_n187_IAFKQ5FUCCT4CGGK3OJ3EFAYF6QMC752.pdh5'
  pdf = DataFrameHDF(rfn, use_dask=True, build_references=True, use_mock_references=True)

  pdf.dataframe = pdf.dataframe.iloc[0:100:2]

  wfn = pdf.to_pdh5('test.pdh5')

  pdf_new = DataFrameHDF(wfn)

  ipdb.set_trace()

### <Mar 14, 2019: use_mock_references speed test>
# df_signal = dataTools.DataFrameHDF(params['/config/sources/signal_sourceDataframePath'], use_dask=True, build_references=True, use_mock_references=True)
# df_signal = dataTools.DataFrameHDF(params['/config/sources/signal_sourceDataframePath'], use_dask=True, build_references=True, use_mock_references=False)
# ipdb.set_trace()
### <Mar 14, 2019: use_mock_references speed test>

if __name__ == '__main__':
  fire.Fire({
              'encode_labels_to_ints': encode_labels_to_ints,
              'stack_dataframehdf': stack_dataframehdf,
              'run_tests': run_tests,
            })
