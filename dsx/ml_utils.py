from dsx.ds_utils import *
from scipy import stats
from sklearn import preprocessing

@pd.api.extensions.register_dataframe_accessor("ml")
class mlx(object):
	'''
	This is an extension for Pandas DataFrame via the "ml" accessor
	'''
	def __init__(self, pandas_obj):
		self._obj = pandas_obj

	# region << Methods >>
	def get_features_categorical(self, cols_ignore: list = None, unique_value_threshold: float = 1.00, excl_by_keywords: list = None) -> list:
		'''
		Get categorical features from the DataFrame.

		:param cols_ignore: column names that will be excluded from the categorical features
		:return: a list of column names that are categorical: list
		'''
		mt = self._obj.ds.meta()
		#mt = df_get_metadata(self._obj)
		if not isinstance(cols_ignore, Iterable):
			if cols_ignore is not None:
				cols_ignore = [cols_ignore]
			else:
				cols_ignore = ['', '']

		# Mainline
		cols = [col for col in mt[(mt.Data_Type == 'object') & (mt.Prcent_Unique_Values < unique_value_threshold)].Col_Name.tolist()
			        if col not in cols_ignore]

		# If keywords provided, loop through list of keywords to remove columns containing the keywords
		if excl_by_keywords is None:
			return cols
		else:
			if not isinstance(excl_by_keywords, Iterable):
				excl_by_keywords = [excl_by_keywords]

			col_list = cols.copy()
			for keyword in excl_by_keywords:
				col_list = [c for c in col_list if keyword not in c]
			return col_list

	def get_features_numerical(self, cols_ignore: list = None, unique_value_threshold: float = 1.00, excl_by_keywords: list = None) -> list:
		'''
		Get numerical features from the DataFrame.
		:param cols_ignore: column names that will be excluded from the categorical features
		:return: a list of column names that are numerical: list
		'''
		mt = self._obj.ds.meta()
		#mt = df_get_metadata(self._obj)
		if not isinstance(cols_ignore, Iterable):
			if cols_ignore is not None:
				cols_ignore = [cols_ignore]
			else:
				cols_ignore = ['', '']

		# Mainline
		cols =  [col for col in mt[(mt.Data_Type != 'object') & (mt.Prcent_Unique_Values < unique_value_threshold)].Col_Name.tolist()
		        if col not in cols_ignore]

		# If keywords provided, loop through list of keywords to remove columns containing the keywords
		if excl_by_keywords is None:
			return cols
		else:
			if not isinstance(excl_by_keywords, Iterable):
				excl_by_keywords = [excl_by_keywords]

			col_list = cols.copy()
			for keyword in excl_by_keywords:
				col_list = [c for c in col_list if keyword not in c]
			return col_list

	def extend_w_dummy_vars(self, columns: list, return_encoder=False) -> pd.core.frame.DataFrame:
		'''
		Generate a full DataFrame by extending dummy variables to the existing DataFrame.
		The method automatically use "K-1 unique value" approach when creating dummy variables for a column.
		:param columns: list of categorical columns
		:param inplace:
		:return:
		'''

		df = self._obj.copy()

		encoder_labelbinarizer = preprocessing.LabelBinarizer()
		for col in columns:
			# encoder_labelbinarizer = preprocessing.LabelBinarizer()
			encoded_labels = encoder_labelbinarizer.fit_transform(df[col])
			encoded_df = pd.DataFrame(encoded_labels, index=df.index)
			if len(encoded_df.columns) == 1:
				encoded_df.columns = [col + '_' + encoder_labelbinarizer.classes_[1]]
			elif len(encoded_df.columns) >= 2:
				encoded_df.columns = [col + '_' + x for x in encoder_labelbinarizer.classes_]
				encoded_df = encoded_df.iloc[:, :-1].copy()
			df = pd.concat([df, encoded_df], 'columns', ignore_index=False, sort=False)
		matched_bool, n = self._obj.ds.len_compare(df)
		if matched_bool:
			if return_encoder:
				return df.copy()
			else:
				return (df.copy(), encoder_labelbinarizer)


	def explore_categorical(self, columns: list, plotseries_label: str='count', figwidth:int=16) -> dict:
		'''
		Generate a barplot for each of the categorical columns and an accummulated percentage report.
		:param columns: list of categorical columns
		:param plotseries_label: str -> 'count or 'percentage'
		:return: dict: Accumulated Percentage Report. Dictionary keys are the column names
		'''
		fetcher = {}
		for col in columns:
			fetcher[col] = self._obj.ds.cumsum(col)
			p = sns.countplot(self._obj[col])
			p.figure.set_figwidth(figwidth)
			plt.tight_layout()

			if plotseries_label == 'count':
				dsx.plt_labels()
			elif plotseries_label == 'percentage':
				dsx.plt_labels(percent=True, denominator=len(self._obj[self._obj[col].notnull()]))
			plt.show()
			print("Generated Accumulated Percentage Report for Column - " + col)
		return fetcher


	def explore_numerical(self, columns: list, cumulative=False):
		df = self._obj
		self.vix_distribution(columns, cumulative)
		return df[columns].describe().T


	def vix_association_matrix(self, k=-1, target:str=None, annot:bool=True, cmap:str='coolwarm', font_size:int=10, fig_size:int=16):
		plt.clf()
		corr_mat = self._obj.corr()
		if k <= 0:
			g = sns.heatmap(corr_mat, annot=annot, cmap=cmap, annot_kws={'size': font_size})
			g.figure.set_size_inches(fig_size, fig_size)
			plt.xticks(rotation=45)
			plt.show()
		elif (k > 0) & (target is not None):
			cols = corr_mat.nlargest(k, target)[target].index
			cm = data[cols].corr()
			hm = sns.heatmap(cm, annot=annot, square=True, fmt='.2f', cmap=cmap, annot_kws={'size': font_size})
			plt.show()


	def vix_facetgrid(self, col, colname_row, colname_col, plt_type:object=plt.hist):
		plt.clf()
		g = sns.FacetGrid(data=self._obj, row=colname_row, col=colname_col)
		g = g.map(plt_type, col)
		plt.show()


	def vix_distribution(self, columns: Union[list, str], cumulative=False, probplot=False):
		if not isinstance(columns, Iterable):
			columns = [columns]

		for col in columns:
			plt.clf()
			sns.distplot(self._obj[col].dropna(), fit=stats.norm, kde=False,
			             hist_kws=dict(cumulative=cumulative), kde_kws=dict(cumulative=cumulative))
			plt.title(col)
			plt.show()

			if probplot:
				fig = plt.figure()
				res = stats.probplot(self._obj[col].dropna(), plot=plt)
				plt.title(col)
				plt.show()


	def vix_correlation_matrix(self, columns=None):
		plt.clf()
		if columns is None:
			corr_mat = self._obj.corr()
		else:
			corr_mat = self._obj[columns].corr()
		g = sns.heatmap(corr_mat, annot=True, cmap='coolwarm', annot_kws={'size': 10})
		g.figure.set_size_inches(16, 16)
		plt.xticks(rotation=45)
		plt.show()


	def vix_correlation_matrix_top_target(self, target:str, columns:list=None, top: int = 7):
		k = top
		if columns is None:
			corr_mat = self._obj.corr()
		else:
			corr_mat = self._obj[columns].corr()
		cols = corr_mat.nlargest(k, target)[target].index
		plt.clf()
		cm = self._obj[cols].corr()
		hm = sns.heatmap(cm, annot=True, square=True, fmt='.1f', cmap='coolwarm', annot_kws={'size': 10})
		plt.show()
	# endregion << Methods >>

	# region << Static Methods >>
	@staticmethod
	def vix_curve_precisions_recalls_thresholds(recalls, precisions, thresholds):
		plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
		plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
		plt.xlabel("Threshold")
		plt.legend(loc="upper left")
		plt.ylim([0, 1])
		plt.show()

	@staticmethod
	def vix_curve_precisions_recalls(recalls, precisions):
		plt.clf()
		plt.plot(precisions, recalls, "b-")
		plt.xlabel("Precisions")
		plt.ylabel("Recalls")
		plt.show()

	@staticmethod
	def vix_curve_truepositive_falsepositive(fpr, tpr, label=None):
		plt.plot(fpr, tpr, linewidth=2, label=label)
		plt.plot([0, 1], [0, 1], 'k--')
		plt.axis([0, 1, 0, 1])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.show()

	@staticmethod
	def vix_features_importance(features, clf) -> pd.core.frame.DataFrame:
		df = pd.DataFrame(features)
		df['Importance'] = clf.feature_importances_
		df.columns = ['Feature', 'Importance']
		df.sort_values(['Importance'], 0, False, True)
		p =  df.plot.bar('Feature', 'Importance', figsize=(16,9))
		plt.show()
		return df

	@staticmethod
	def vix_cofficient(features, reg) -> pd.core.frame.DataFrame:
		coef_df = pd.DataFrame(features)
		coef_df['Coefficient'] = reg.coef_
		coef_df.columns = ['Variables', 'Coefficient']
		temp = pd.DataFrame([['Intercept', reg.intercept_]], columns=['Variables', 'Coefficient'])
		coef_df = coef_df.append(temp)
		coef_df.sort_values(['Coefficient'], 0, False, True)
		p = coef_df[coef_df.Variables != 'Intercept'].plot.bar('Variables', 'Coefficient', figsize=(16, 9))
		plt.show()
		return coef_df
	# endregion << Static Methods >>