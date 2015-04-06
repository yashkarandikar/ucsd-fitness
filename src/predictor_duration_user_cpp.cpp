#include<string>
#include<fstream>
#include<iostream>
#include<stdlib.h>
#include<sstream>
#include<algorithm>
#include<random>
#include<chrono>
#include<vector>
#include<cassert>
#include<lbfgs.h>
#include <dlib/matrix.h>
#include <dlib/optimization.h>
#include<cmath>

using namespace std;
typedef dlib::matrix<double,0,1> column_vector;

double** alloc_matrix(int N, int M);
void free_matrix(double **mat, int nrows, int ncols);

class Matrix
{
    public:
        int shape[2];
        double **data;

        Matrix(int nrows, int ncols, bool allocate = true)
        {
            if (allocate) 
                data = alloc_matrix(nrows, ncols);
            shape[0] = nrows;
            shape[1] = ncols;
        }

        double* operator[](int i)
        {
            return data[i];
        }

        void free()
        {
            free_matrix(data, shape[0], shape[1]);
        }
};

double** alloc_matrix(int N, int M)
{
    double **mat = new double*[N];
    for (int i = 0; i < N; i++)
        mat[i] = new double[M];
    return mat;
}

void free_matrix(double **mat, int nrows, int ncols)
{
    for (int i = 0 ; i < nrows; i++) {
        delete[] mat[i];
    }
    delete[] mat;
}

double** zeros(int nrows, int ncols)
{
    double **z = alloc_matrix(nrows, ncols);
    for (int i = 0; i < nrows; i++) {
        for (int j = 0 ; j < ncols; j++) {
            z[i][j] = 0;
        }
    }
    return z;
}

Matrix zeros_matrix(int nrows, int ncols)
{
    Matrix m(nrows, ncols);
    for (int i = 0 ; i < nrows; i++) {
        for (int j = 0 ; j < ncols; j++) {
            m.data[i][j] = 0;
        }
    }
    return m;
}

Matrix read_matrix(const char *infile, int& N)
{
    // count lines
    N = 0;
    ifstream f;
    f.open(infile);
    string line;
    string token;
    int ncols = 0;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        if (ncols == 0) {
            while (iss >> token)
                ncols++;
        }
        N++;
    }
    cout << "Data matrix dimensions = " << N << " x " << ncols << "\n";
    f.close();

    f.open(infile);
    Matrix data(N, ncols);
    int i = 0;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        for (int j = 0 ; j < ncols; j++) {
            iss >> token;
            data[i][j] = stod(token);
        }
        i++;
    }
    f.close();
    return data;
}

void init_random(double **a, int n, int m)
{
    for (int i = 0 ; i < n; i++)
        for (int j = 0 ; j < m; j++)
            a[i][j] = ((double)rand()/(double)RAND_MAX);
}

void init_random(double* a, int n)
{
    for (int i = 0; i < n; i++)
        a[i] = ((double)rand()/(double)RAND_MAX);
}

void clear(double *a, int n)
{
    for (int i = 0; i < n; i++)
        a[i] = 0;
}

void clear(vector<double>& v)
{
    clear(&v[0], v.size());
}

int randint(int low, int high)
{
    int range = high - low + 1;
    return (low + (rand() % range));
}

int get_user_count(Matrix& data) 
{
    int nrows = data.shape[0];
    return (int)(data[nrows-1][0] + 1);
}

class Constants
{
    public:
        Matrix* data;
        double lam;
        vector<int> *u_indices;
};

double F(const lbfgsfloatval_t *theta, Matrix& data, double lam)
{
    int i = 0;
    int N = data.shape[0];
    int U = get_user_count(data);
    int nparams = U + 3;
    double e = 0, alpha, d, t, temp;
    double alpha_all = theta[nparams - 3];
    double theta_0 = theta[nparams - 2];
    double theta_1 = theta[nparams - 1];
    int u;
    while (i < N) {
        u = data[i][0];
        alpha = theta[u];
        while (i < N && data[i][0] == u) {
            d = data[i][2];
            t = data[i][3];
            temp = (alpha + alpha_all) * (theta_0 + theta_1 * d) - t ;
            e += temp * temp;
            i += 1;
        }
    }
    // regularization
    //e += lam * theta[:-3].dot(theta[:-3])
    double reg = 0;
    for (int i = 0; i < nparams - 3; i++) {
        reg += theta[i] * theta[i];
    }
    e += lam * reg;
    e = e / (double) N;
    return e;
}

vector<int> generate_u_indices(Matrix& data, int U)
{
    int N = data.shape[0];
    vector<int> col0(N);
    for (int i = 0 ; i < N; i++)
        col0[i] = data[i][0];
    vector<int> u_indices(U+1);
    for (int u = 0 ; u < U; u++) {
        u_indices[u] = lower_bound(col0.begin(), col0.end(), u) - col0.begin();
    }
    u_indices[U] = N;
    return u_indices;
}

void Fprime(const lbfgsfloatval_t *theta, Matrix& data, double lam, double *dE, vector<int>& u_indices)
{
    int N = data.shape[0];
    int U = get_user_count(data);
    int nparams = U + 3;
    double alpha = theta[nparams - 3];
    double theta_0 = theta[nparams - 2];
    double theta_1 = theta[nparams - 1];
    clear(dE, nparams);

    //np.ndarray col0 = np.ravel(data[:, 0])
    //np.ndarray uins = np.array(range(0, n_users))
    //u_indices = list(np.searchsorted(col0, uins))
    //u_indices.append(N)
        
    double dE_theta0 = 0.0;
    double dE_theta1 = 0.0;

    int start_u, end_u;
    double alpha_u, t, d, t0_t1_d, dE0, tpred;

    for (int i = 0; i < U; i++) {
        start_u = u_indices[i];
        end_u = u_indices[i+1];
        alpha_u = theta[i];
        for (int j = start_u; j < end_u; j++) {
            assert(data[j][0] == i);
            t = data[j][3];
            d = data[j][2];
            
            t0_t1_d = theta_0 + theta_1 * d;

            tpred = (alpha_u + alpha) * t0_t1_d;
            dE[i] += 2 * (tpred - t) * t0_t1_d;
            dE[nparams - 3] += 2 * (tpred - t) * t0_t1_d;
            
            //a_t0_t1_d = alpha_u * t0_t1_d;
            //#dE[i] = dE[i] + 2 * (a_t0_t1_d - t) * t0_t1_d;

            //dE / d_theta_0 and 1
            dE0 = 2 * (alpha_u + alpha) * (tpred - t);
            dE_theta0 += dE0;
            dE_theta1 += dE0 * d;
        }
    }
        
    dE[nparams - 2] = dE_theta0;
    dE[nparams - 1] = dE_theta1;

    // regularization
    // dE = dE + lam * np.multiply(dE, (2 * theta))
    for (int u = 0; u < U; u++)
        dE[u] += 2 * lam * theta[u];

    for (int i = 0 ; i < nparams; i++)
        dE[i] = dE[i] / (double) N;
}

static lbfgsfloatval_t evaluate(void *instance, const lbfgsfloatval_t *theta, lbfgsfloatval_t *g, const int n, const lbfgsfloatval_t step)
{
    lbfgsfloatval_t fx = 0.0;
    Constants *consts = (Constants*) instance;
    //vector<double> theta(x, x + consts -> nparams);
    fx =  F(theta, *(consts -> data), consts -> lam);
    Fprime(theta, *(consts -> data), consts -> lam, g, *(consts -> u_indices));
    return fx;
}

static int progress(void *instance, const lbfgsfloatval_t *x, const lbfgsfloatval_t *g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n, int k,int ls )
{
    printf("Iteration %d: \n", k);
    printf("@fx = %f\n", fx);
    //printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    //printf("\n");
    return 0;
}

void optimize(lbfgsfloatval_t* theta, Matrix& data, double lam, lbfgs_parameter_t& param)
{
    lbfgsfloatval_t fx = 0;
    int U = get_user_count(data);
    Constants c;
    c.data = &data;
    c.lam = lam;
    vector<int> u_indices = generate_u_indices(data, U);
    c.u_indices = &u_indices;
    int ret = lbfgs(U + 3, theta, &fx, evaluate, progress, (void *)&c, &param);
    printf("LBFGS terminated with status %d\n", ret);
}

double F_dlib(Constants c, const column_vector x)
{
    lbfgsfloatval_t *theta = new lbfgsfloatval_t[x.nr()];
    for (int i = 0 ; i < x.nr(); i++)
        theta[i] = x(i, 0);
    double f = F(theta, *(c.data), c.lam);
    delete[] theta;
    return f;
}

void learn(char *infile, double lam, char* outfile)
{
    int N;
    Matrix data = read_matrix(infile, N);
    int U = get_user_count(data);
    cout << "U = " << U << "\n";
    int nparams = U + 3;
    lbfgsfloatval_t *theta = lbfgs_malloc(nparams);
    init_random(theta, nparams);
    lbfgs_parameter_t lbfgsparam;
    lbfgs_parameter_init(&lbfgsparam);
    lbfgsparam.epsilon = 1e-8;
    lbfgsparam.m = 10;
    bool check_grad = false;

    if (check_grad) {
        printf("Checking gradient before training..\n");
        column_vector m(nparams);
        for (int i = 0 ; i < nparams; i++) {
            m(i, 0) = theta[i];
        }

        // Analytic
        //ourgrad = np.linalg.norm(eprime_fn(np.array(theta), train, lam), ord = 2)
        //print "gradient = ", ourgrad
        lbfgsfloatval_t *dE = new lbfgsfloatval_t[nparams];
        vector<int> u_indices = generate_u_indices(data, U);
        Fprime(theta, data, lam, dE, u_indices);
        double analytic = 0;
        for (int i = 0 ; i < nparams; i++) {
            analytic += dE[i] * dE[i];
        }
        analytic = sqrt(analytic);
        cout << "analytic = " << analytic << "\n";
        delete[] dE;

        // Numerical
        Constants c;
        c.data = &data;
        c.lam = lam;
        column_vector numerical_grad_vec = dlib::derivative(F_dlib)(c, m);
        double numerical = 0;
        for (int i = 0; i < nparams; i++) {
            double vi = numerical_grad_vec(i, 0);
            numerical += vi * vi;
        }
        numerical = sqrt(numerical);
        cout << "numerical = " << numerical << "\n";

        double ratio = analytic / numerical;
        printf("Ratio = %lf\n", ratio);
        cout << "Ratio = " << ratio << "\n";
        cout << fabs(1.0 - ratio) << "\n";
        assert(fabs(1.0 - ratio) < 1e-6);
        exit(0);
    }

    optimize(theta, data, lam, lbfgsparam);

    ofstream of;
    of.open(outfile);
    of << "theta=[";
    int i = 0;
    for (i = 0 ; i < nparams - 1; i++)
        of << theta[i] << ",";
    of << theta[i] << "]\n";
    of.close();
}

int main(int argc, char* argv[])
{
    srand(12345);
    // arguments will have data file name, lam1, lam2, out file name
    assert(argc == 4);
    char* infile = argv[1];
    double lam1 = atof(argv[2]);
    char* outfile = argv[3];
    learn(infile, lam1, outfile);
    return 0;
}
