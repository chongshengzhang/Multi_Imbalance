#include <stdio.h>
#include <ctype.h>

#include "mex.h"

#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif 

void exit_with_help()
{
	mexPrintf(
	"Usage: [label_vector, instance_matrix] = read_sparse(fname);\n"
	);
}

static void fake_answer(mxArray *plhs[])
{
	plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(0, 0, mxREAL);
}

/* read in a problem (in svmlight format) */
void read_problem(const char *filename, mxArray *plhs[])
{
	int elements, max_index, min_index, i, k;
	FILE *fp = fopen(filename,"r");
	int l = 0;
	mwIndex *ir, *jc;
	double *labels, *samples;
	
	if(fp == NULL)
	{
		mexPrintf("can't open input file %s\n",filename);
		fake_answer(plhs);
		return;
	}

	max_index = 0;
	min_index = 1; /* our index starts from 1 */
	elements = 0;
	while(1)
	{
		/* label */
		int index;
		double value;
		fscanf(fp,"%lf",&value);

		/* features */
		while(1)
		{
			int c;
			do {
				c = getc(fp);
				if(c=='\n') goto out;
				if(c==EOF) goto eof;
			} while(isspace(c));
			ungetc(c,fp);
			fscanf(fp,"%d:%lf",&index, &value);
			if (index < min_index)
				min_index = index;
			elements++;
		}	
out:
		if(index > max_index)
			max_index = index;
		l++;
	}
eof:
	rewind(fp);

	/* y */
	plhs[0] = mxCreateDoubleMatrix(l, 1, mxREAL);
	/* x^T */
	if (min_index <= 0)
		plhs[1] = mxCreateSparse(max_index-min_index+1, l, elements, mxREAL);
	else
		plhs[1] = mxCreateSparse(max_index, l, elements, mxREAL);

	labels = mxGetPr(plhs[0]);
	samples = mxGetPr(plhs[1]);
	ir = mxGetIr(plhs[1]);
	jc = mxGetJc(plhs[1]);

	k=0;
	for(i=0;i<l;i++)
	{
		jc[i] = k;
		fscanf(fp,"%lf",&labels[i]);

		while(1)
		{
			int c, index;
			do {
				c = getc(fp);
				if(c=='\n') goto out2;
			} while(isspace(c));
			ungetc(c,fp);
			fscanf(fp,"%d:%lf",&index,&samples[k]);
			ir[k] = index - min_index; /* precomputed kernel has <index> start from 0 */
			++k;
		}	
out2:
		;
	}
	jc[l] = k;

	fclose(fp);

	{
		mxArray *rhs[1], *lhs[1];
		rhs[0] = plhs[1];
		if(mexCallMATLAB(1, lhs, 1, rhs, "transpose"))
		{
			mexPrintf("Error: cannot transpose problem\n");
			return;
		}
		plhs[1] = lhs[0];
	}
}

void mexFunction( int nlhs, mxArray *plhs[],
		int nrhs, const mxArray *prhs[] )
{
	if(nrhs == 1)
	{
		char filename[256];

		mxGetString(prhs[0], filename, mxGetN(prhs[0]) + 1);

		if(filename == NULL)
		{
			mexPrintf("Error: filename is NULL\n");
			return;
		}

		read_problem(filename, plhs);
	}
	else
	{
		exit_with_help();
		fake_answer(plhs);
		return;
	}
}
