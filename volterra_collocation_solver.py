import numpy as np
from scipy.interpolate import lagrange

def lagrange_poly(node_indx, nodes):
    return lagrange(nodes, [1.0 if (i == node_indx) else 0 for i,_ in enumerate(nodes)])

def integ_lpoly(node_indx, nodes):
    return lagrange_poly(node_indx, nodes).integ()

def A(coll_info):
    coll_params = coll_info.params
    num_coll_params = len(coll_params)

    matrix = np.zeros((num_coll_params, num_coll_params))
    for i,j in np.ndindex(matrix.shape):
        integ_at_zero = integ_lpoly(j, coll_params)(0.0)
        integ_at_c_i = integ_lpoly(j, coll_params)(coll_params[i])
        matrix[i,j] = integ_at_c_i - integ_at_zero
    return matrix

def a(n, a_data, coll_info):
    coll_divs = coll_info.divs
    coll_choices = coll_info.choices
    an_indices = [n * coll_divs**2 + c * coll_divs for c in coll_choices]
    return np.array(a_data[an_indices])

def An(mesh_indx_n, a_data, coll_info):
    return np.matmul(np.diag(a(mesh_indx_n, a_data, coll_info)), A(coll_info))

def CNL(mesh_indx_n, mesh_indx_ell, kernel_data, coll_info):
    assert mesh_indx_ell >= 0, "ell must be non-negative"
    assert mesh_indx_ell < mesh_indx_n, "ell must be smaller than n"
    coll_params = coll_info.params
    num_coll_params = len(coll_params)
    coll_choices = coll_info.choices
    coll_divs = coll_info.divs
    b = quad_weights(coll_info.params)

    matrix = np.zeros((num_coll_params, num_coll_params))
    betas = [integ_lpoly(j, coll_params) for j in range(num_coll_params)]
    for i,j in np.ndindex(matrix.shape):
        for k in range(num_coll_params):     
            sub_indx = (coll_choices[i] - coll_choices[k]) * coll_divs
            kern_indx = (mesh_indx_n - mesh_indx_ell) * coll_divs**2 + sub_indx
            matrix[i,j] += b[k]*kernel_data[kern_indx]*betas[j](coll_params[k])
    return matrix

def CN(kernel_data, coll_info):
    coll_choices = coll_info.choices
    coll_params = coll_info.params
    num_coll_params = len(coll_params)
    coll_divs = coll_info.divs
    b = quad_weights(coll_params)

    matrix = np.zeros((num_coll_params, num_coll_params))
    betas = [integ_lpoly(j, coll_params) for j in range(num_coll_params)]
    c = coll_params
    for i,j in np.ndindex(matrix.shape):
        for k in range(num_coll_params):
            kern_indx = coll_choices[i]*coll_divs - coll_choices[i]*coll_choices[k]        
            matrix[i,j] += c[i]*b[k] * kernel_data[kern_indx] * betas[j](c[i] * c[k])
    return matrix

def kappa_n(mesh_indx_n, kernel_data, a_data, coll_info, dt):
    coll_divs = coll_info.divs
    coll_choices = coll_info.choices
    coll_params = coll_info.params
    num_coll_params = len(coll_params)
    b = quad_weights(coll_params)

    vector = np.zeros((num_coll_params))
    for i in range(num_coll_params):
        for k in range(num_coll_params):
            kern_indx = coll_choices[i]*coll_divs - coll_choices[i]*coll_choices[k]        
            vector[i] += b[k] * kernel_data[kern_indx]
        vector[i] *= coll_params[i]
    return a(mesh_indx_n, a_data, coll_info) + dt * vector

def kappa_nl(mesh_indx_n, mesh_indx_ell, kernel_data, coll_info):
    assert mesh_indx_ell >= 0, "ell must be non-negative"
    assert mesh_indx_ell < mesh_indx_n, "ell must be smaller than n"

    coll_divs = coll_info.divs
    coll_choices = coll_info.choices
    coll_params = coll_info.params
    num_coll_params = len(coll_params)
    b = quad_weights(coll_params)

    ans_vec = np.zeros(num_coll_params)
    for i in range(num_coll_params):
        for k in range(num_coll_params):
            sub_indx = (coll_choices[i] - coll_choices[k]) * coll_divs
            kern_indx = (mesh_indx_n - mesh_indx_ell) * coll_divs**2 + sub_indx
            ans_vec[i] += b[k] * kernel_data[kern_indx]
    return ans_vec

def G_VIDE(mesh_indx_n, current_solution, boundary_vals, kernel_data, coll_info, dt):
    num_coll_params = len(coll_info.params)
    y = boundary_vals
    big_y = current_solution
    vector = np.zeros(num_coll_params)
    for ell in range(mesh_indx_n):
        vector += dt * boundary_vals[ell] * kappa_nl(mesh_indx_n, ell, kernel_data, coll_info)
        vector += dt**2 * np.matmul(CNL(mesh_indx_n, ell, kernel_data, coll_info), big_y[ell,:])
    return vector 

def BNL(mesh_indx_n, mesh_indx_ell, kernel_data, coll_info):
    assert mesh_indx_ell >= 0, "ell must be non-negative"
    assert mesh_indx_ell < mesh_indx_n, "ell must be smaller than n"
    num_coll_params = len(coll_info.choices)
    weights = quad_weights(coll_info.params)
    coll_divs = coll_info.divs
    coll_choices = coll_info.choices
    matrix = np.zeros((num_coll_params, num_coll_params))
    for i,j in np.ndindex(matrix.shape):
        mesh_point_indx = (mesh_indx_n - mesh_indx_ell) * coll_divs**2
        sub_indx = (coll_choices[i] - coll_choices[j]) * coll_divs
        matrix[i,j] = weights[j]*kernel_data[mesh_point_indx + sub_indx]
    return matrix

def BN(kernel_data, coll_info, add_zero_node=False):
    num_coll_params = len(coll_info.choices)
    b = quad_weights(coll_info.params)
    c = coll_info.params
    coll_divs = coll_info.divs
    coll_choices = coll_info.choices
    coll_params = coll_info.params

    matrix = np.zeros((num_coll_params, num_coll_params))
    # polys = [lagrange_poly(j, coll_params) for j in range(num_coll_params)]
    if add_zero_node:
        nodes = [0] + list(coll_params)
        polys = [lagrange_poly(j+1, nodes) for j in range(num_coll_params)]
    else:
        polys = [lagrange_poly(j, coll_params) for j in range(num_coll_params)]
    for i,j in np.ndindex(matrix.shape):
        for k in range(num_coll_params):
            k_indx = coll_choices[i]*coll_divs - coll_choices[i]*coll_choices[k]
            matrix[i,j] += c[i]*b[k] * kernel_data[k_indx] * polys[j](c[i] * c[k])
    return matrix

def G(mesh_indx_n, current_solution, kernel_data, coll_info, dt):
    num_coll_params = len(coll_info.choices)
    vector = np.zeros((num_coll_params))
    for ell in range(mesh_indx_n):
        vector += dt * np.matmul(BNL(mesh_indx_n, ell, kernel_data, coll_info), current_solution[ell,:])
    return vector

def rho(mesh_indx_n, kernel_data, coll_info):
    coll_divs = coll_info.divs
    coll_params = coll_info.params
    coll_choices = coll_info.choices
    num_coll_params = len(coll_params)

    nodes = [0] + list(coll_params)
    lpoly = lagrange_poly(0, nodes)
    vector = np.zeros((num_coll_params))
    for i in range(num_coll_params):
        for k in range(num_coll_params):
            kernel_indx = coll_choices[i]*coll_divs - coll_choices[i]*coll_choices[k]
            c = coll_params
            b = quad_weights(coll_params)
            vector[i] += c[i]*b[k] * kernel_data[kernel_indx] * lpoly(c[i] * c[k])
    return -vector

class CollInfo: 
    def __init__(self, divs, choices) -> None:
        self.divs = divs
        self.choices = choices
        self.params = np.array([i*(1.0/divs) for i in choices])
        self.weights = quad_weights(self.params)

def quad_weights(coll_params):
    num_coll_params = len(coll_params)
    weights = np.zeros((num_coll_params))
    for j in range(num_coll_params):
        integral_at_zero = lagrange_poly(j, coll_params).integ()(0.0)
        integral_at_one = lagrange_poly(j, coll_params).integ()(1.0)
        weights[j] = integral_at_one - integral_at_zero
    return weights

def g(n, g_data, coll_info):
    coll_divs = coll_info.divs
    coll_choices = coll_info.choices
    gn_indices = [n * coll_divs**2 + c * coll_divs for c in coll_choices]
    return np.array(g_data[gn_indices])

def poly_piece(mesh_indx, solution_U, coll_info, init_val=None):
    nodes = [0] + list(coll_info.params)
    def poly(rel_x):
        if init_val is not None:
            value = init_val * lagrange_poly(0, nodes)(rel_x)
            for i, u in enumerate(solution_U[mesh_indx]):
                value += u * lagrange_poly(i+1, nodes)(rel_x)
        else:
            value = 0.0
            for i, u in enumerate(solution_U[mesh_indx]):
                value += u * lagrange_poly(i, coll_info.params)(rel_x)
        return value
    return poly

# def poly_piece(mesh_indx, solution_U, coll_info, init_val=None):
#     nodes = [0] + list(coll_info.params)
#     if init_val is not None:
#         poly = init_val * lagrange_poly(0, nodes)
#         for i, u in enumerate(solution_U[mesh_indx]):
#             poly += u * lagrange_poly(i+1, nodes)
#     else:
#         poly = np.polynomial.Polynomial([0])
#         for i, u in enumerate(solution_U[mesh_indx]):
#             poly += np.polynomial.Polynomial([u]) * lagrange_poly(i, coll_info.params)
#     return poly

def poly_piece_VIDE(mesh_indx, solution_Y, coll_info, init_value, dt):
    coll_params = coll_info.params
    num_coll_params = len(coll_params)
    betas = [integ_lpoly(j, coll_params) for j in range(num_coll_params)]
    def poly(rel_x):
        value = init_value
        for indx, y in enumerate(solution_Y[mesh_indx]):
            value += dt * y * betas[indx](rel_x)
        return value
    return poly

def solve_VIDE(*, g_values, kernel_values, a_values, soln_init_value, time_step,
               coll_divs=2, coll_choices=[0,1,2], return_polys=False):
    '''
    Solves the following Volterra integro-differential equation (VIDE) for the
    unknown function y(t). 

    y'(t) = a(t)*y(t) + g(t) + integral[K(t-s)y(s)ds from s=0 to s=t]

    Returns a two element tuple (soln_values, polys) where
    soln_values is a list of y-values and polys is a list of the polynomial

            Parameters:
                    a (int): A decimal integer
                    b (int): Another decimal integer

            Returns:
                    binary_sum (str): Binary string of the sum of a and b
    '''
    assert g_values.shape == kernel_values.shape
    assert a_values.shape == kernel_values.shape
    assert len(kernel_values.shape) == 1

    coll_info = CollInfo(coll_divs, coll_choices)
    num_coll_params = len(coll_info.params)
    dt = time_step * coll_divs**2

    assert (len(kernel_values) - 1) % coll_divs**2 == 0
    mesh_divs = int((len(kernel_values)-1) / coll_divs**2)
    num_mesh_points = mesh_divs + 1

    solution_Y = np.zeros((mesh_divs , num_coll_params))
    boundary_vals = np.zeros((num_mesh_points))
    boundary_vals[0] = soln_init_value

    for n in range(mesh_divs):
        rhs_vector = g(n, g_values, coll_info) \
            + G_VIDE(n, solution_Y, boundary_vals, kernel_values, coll_info, dt) \
            + boundary_vals[n]*kappa_n(n, kernel_values, a_values, coll_info, dt)
        coef_matrix = np.identity(num_coll_params)  \
                        - dt*(An(n, a_values, coll_info) \
                    + dt*CN(kernel_values, coll_info))
        solution_Y[n] = np.linalg.solve(coef_matrix, rhs_vector)
        # next_mesh_time = data_times[(n+1)*coll_divs**2]
        boundary_vals[n+1] = poly_piece_VIDE(n, solution_Y, coll_info, boundary_vals[n], dt)(1.0)

    soln_values = np.zeros_like(g_values)
    soln_polys = []
    for n in range(mesh_divs):
        poly = poly_piece_VIDE(n, solution_Y, coll_info, boundary_vals[n], dt)
        soln_polys.append(poly)
        for i in range(coll_divs**2 + 1):
            soln_values[n*coll_divs**2 + i] += poly(i*(1.0/coll_divs**2))
        
    # At each mesh point (other than the first and last), we have added the value of 
    # the two adjacent polynomials. Now, we turn this into the average.
    for n in range(1, mesh_divs):
        soln_values[n*coll_divs**2] *= 0.5

    if return_polys:
        return (soln_values, soln_polys)
    return soln_values


def solve_VIE_1(*, 
                g_values, 
                kernel_values,
                soln_init_value, 
                time_step, 
                coll_divs=3, 
                coll_choices=[1,2,3], 
                return_polys=False, 
                force_continuous=False):
    assert g_values.shape == kernel_values.shape
    assert len(g_values.shape) == 1
    assert 0 not in coll_choices

    coll_info = CollInfo(coll_divs, coll_choices)
    num_coll_params = len(coll_info.params)
    dt = time_step * coll_divs**2

    assert (len(kernel_values) - 1) % coll_divs**2 == 0
    mesh_divs = int((len(kernel_values)-1) / coll_divs**2)
    num_mesh_points = mesh_divs + 1

    solution_U = np.zeros((mesh_divs, num_coll_params))
    if not force_continuous:
        for n in range(mesh_divs):
            rhs_vector = g(n, g_values, coll_info) - G(n, solution_U, kernel_values, coll_info, dt)
            coef_matrix = dt*BN(kernel_values, coll_info)
            solution_U[n] = np.linalg.solve(coef_matrix, rhs_vector)
    else:
        boundary_vals = np.zeros((num_mesh_points))
        boundary_vals[0] = soln_init_value
        for n in range(mesh_divs):
            rhs_vector = g(n, g_values, coll_info) \
                - G(n, solution_U, kernel_values, coll_info, dt) \
                + dt*boundary_vals[n]*rho(n, kernel_values, coll_info)
            coef_matrix = dt*BN(kernel_values, coll_info, add_zero_node=True)
            solution_U[n] = np.linalg.solve(coef_matrix, rhs_vector)
            # next_mesh_time = data_times[(n+1)*coll_divs**2]
            boundary_vals[n+1] = poly_piece(n, solution_U, coll_info, boundary_vals[n])(1.0)
    
    soln_values = np.zeros_like(g_values)
    soln_polys = []
    for n in range(mesh_divs):
        if force_continuous:
            poly = poly_piece(n, solution_U, coll_info, boundary_vals[n])
        else:
            poly = poly_piece(n, solution_U, coll_info)
        soln_polys.append(poly)
        for i in range(coll_divs**2 + 1):
            soln_values[n*coll_divs**2 + i] += poly(i*(1.0/coll_divs**2))

    # At each mesh point (other than the first and last), we have added the value of 
    # the two adjacent polynomials. Now, we turn this into the average.
    for n in range(1, mesh_divs):
        soln_values[n*coll_divs**2] *= 0.5

    if return_polys:
        return (soln_values, soln_polys)
    return soln_values
    


def solve_VIE_2(*, 
                g_values, 
                kernel_values, 
                time_step, 
                coll_divs=2, 
                coll_choices=[0,1,2], 
                return_polys=False):
    assert g_values.shape == kernel_values.shape
    assert len(g_values.shape) == 1

    coll_info = CollInfo(coll_divs, coll_choices)
    num_coll_params = len(coll_choices)
    dt = time_step * coll_divs**2

    assert (len(kernel_values) - 1) % coll_divs**2 == 0
    mesh_divs = int((len(kernel_values)-1) / coll_divs**2)

    solution_U = np.zeros((mesh_divs, num_coll_params))
    for n in range(mesh_divs):
        coef_matrix = np.identity(num_coll_params) - dt * BN(kernel_values, coll_info)
        rhs_vector = g(n, g_values, coll_info) + G(n, solution_U, kernel_values, coll_info, dt)
        solution_U[n] = np.linalg.solve(coef_matrix, rhs_vector)

    soln_values = np.zeros_like(g_values)
    soln_polys = []
    for n in range(mesh_divs):
        poly = poly_piece(n, solution_U , coll_info)
        soln_polys.append(poly)
        for i in range(coll_divs**2 + 1):
            soln_values[n*coll_divs**2 + i] += poly(i*(1.0/coll_divs**2))
        
    # At each mesh point (other than the first and last), we have added the value of 
    # the two adjacent polynomials. Now, we turn this into the average.
    for n in range(1, mesh_divs):
        soln_values[n*coll_divs**2] *= 0.5

    if return_polys:
        return (soln_values, soln_polys)
    return soln_values
