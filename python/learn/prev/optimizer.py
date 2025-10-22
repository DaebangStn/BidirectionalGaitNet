from eo.learn.preprocess import gt_from_exp_name
from eo.learn.module import Regression
from eo.learn.util import *


class Optimizer:
    def __init__(self, ckpt: str, maximize: bool = False, trials: int = 20, regularize: Dict[str, float] = None):
        if "gp" in ckpt:
            from eo.learn.gp.module import load_from_ckpt
            self._model = load_from_ckpt(ckpt).eval()
        else:
            from eo.learn.nn.module import load_from_ckpt
            self._model = load_from_ckpt(ckpt).eval()
        self._device = self._model.device
        self._maximize = maximize
        self._trial = trials
        
        if regularize is not None and len(regularize) > 0:
            regularize_keys = list(regularize.keys())
            assert all(key in self._model.input_col for key in regularize_keys), f"Regularize keys must be in the input columns: {regularize_keys} but model input: {self._model.input_col}"
            self._regularize = regularize
        else:
            self._regularize = None

        gt, stat = gt_from_exp_name(self._model.exp_name, self._model.input_col, self._model.output_col, self._model.transform, is_ckpt=False)
        self._param = {}
        self._param_min = {}
        self._param_max = {}
        for col in self._model.input_col:
            param_min = np.float64(gt.select(pl.col(col).min()).collect().item())
            param_max = np.float64(gt.select(pl.col(col).max()).collect().item())
            self._param[col] = (param_min, param_max)
            mean = np.float64(stat.filter(pl.col("describe") == "mean")[col].item())
            std = np.float64(stat.filter(pl.col("describe") == "std")[col].item())
            self._param_min[col] = param_min * std + mean
            self._param_max[col] = param_max * std + mean

    def _objective(self, x_opt: np.ndarray, idx: int, fixed_fields: Dict[str, float], optimize_fields: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the value and gradient for the scipy.minimize function.
        :param x_opt: 1D array of parameters being optimized
        :param idx: index of the output field to optimize
        :param fixed_fields: dict of field_name to fixed value (normalized values)
        :param optimize_fields: list of field names being optimized
        :return: value and gradient
        """
        # Reconstruct the full input vector
        x_dict = {field: fixed_fields.get(field, None) for field in self._model.input_col}
        for i, field in enumerate(optimize_fields):
            x_dict[field] = float(x_opt[i])
        x_full = torch.tensor([x_dict[field] for field in self._model.input_col], device=self._device, requires_grad=True)
        
        y = self._model(x_full.unsqueeze(0)).squeeze(0)[idx]
        y.backward()
        
        grads = x_full.grad.cpu().detach().numpy()
        grad_opt = np.array([grads[self._model.input_col.index(field)] for field in optimize_fields])

        value = y.cpu().detach().numpy()
        
        if self._regularize is not None:
            for reg_key, reg_weight in self._regularize.items():
                value += reg_weight * x_dict[reg_key]
            # Todo, check the size of grad_opt to add weight for the regularize term
            for opt_field in self._regularize.keys():
                grad_opt[optimize_fields.index(opt_field)] += self._regularize[opt_field]
        
        if self._maximize:
            value = -value
            grad_opt = -grad_opt

        # Ensure float64 for SLSQP compatibility
        return np.float64(value), np.asarray(grad_opt, dtype=np.float64)
       
    def _check_optimized(self, best: Optional[OptimizeResult], found: float) -> bool:
        """
        Checks if the found value is better than the best value
        """
        return (best is None) or (self._maximize and found > best.fun) or (not self._maximize and found < best.fun)

    def run(self, field_name: str, fixed_fields: Dict[str, float] = None) -> Tuple[Dict[str, float], float]:
        """
        Runs the optimization for the given field
        :param field_name: the field to optimize
        :param fixed_fields: dict of field_name to fixed value (not normalized yet)
        :return: the optimized parameters and the expected value
        """
        if fixed_fields is None:
            fixed_fields = {}
        fixed_fields = self.model.normalize(fixed_fields)
        input_fields = self._model.input_col
        optimize_fields = [field for field in input_fields if field not in fixed_fields]
        optimize_idx = self._model.output_col.index(field_name)
        obj = partial(self._objective, idx=optimize_idx, fixed_fields=fixed_fields, optimize_fields=optimize_fields)
        best = None

        for _ in range(self._trial):
            initial_guess = np.array([np.random.uniform(self._param[field][0], self._param[field][1]) for field in optimize_fields])
            bounds = [self._param[field] for field in optimize_fields]
            result = minimize(obj, initial_guess, jac=True, bounds=bounds)
            if self._check_optimized(best, result.fun):
                best = result

        # Reconstruct the full optimized input, including fixed fields
        best_x_full = {field: fixed_fields.get(field, None) for field in input_fields}
        for i, field in enumerate(optimize_fields):
            best_x_full[field] = best.x[i] if best.x.ndim > 0 else best.x
        best_x_full = self.model.denormalize(best_x_full)
        best_return = self.model.denormalize({field_name: best.fun})[field_name]
        return best_x_full, best_return


    def run_w_cfield_from_model(self, field_name: str, c_field: str, c_value: float) -> Tuple[Dict[str, float], float]:
        """
        Runs the optimization for the given field
        :param field_name: the field to optimize (e.g. cot_ma2)
        :param c_field: the field of constraint from the model (e.g. velocity)
        :param c_value: the value of constraint (e.g. 0.5m/s)
        :return: the optimized parameters and the expected value
        """
        c_value_norm = self.model.normalize({c_field: c_value})[c_field]
        c_field_idx = self.model.output_col.index(c_field)
        
        def cfield_constraint_fn(x: np.array) -> float:
            x_tensor = torch.tensor(x, device=self._device, requires_grad=False, dtype=torch.float32)
            x_tensor = x_tensor.unsqueeze(0)
            with torch.no_grad():
                output = self.model(x_tensor)
            c_output = output.squeeze()[c_field_idx]
            value = float(c_output.cpu().numpy() - c_value_norm)
            return np.float64(value)  # Ensure float64 for SLSQP compatibility
            
        def cfield_constraint_grad(x: np.array) -> np.ndarray:
            x_tensor = torch.tensor(x, device=self._device, requires_grad=True, dtype=torch.float32)
            x_tensor = x_tensor.unsqueeze(0)
            output = self.model(x_tensor)
            c_output = output.squeeze()[c_field_idx]
            
            # Create gradient tape and compute gradient w.r.t. input
            grad = torch.autograd.grad(c_output, x_tensor)[0]
            if grad is None:
                print(f"[Warning] Gradient is None for x: {x}")
                grad = torch.zeros_like(x_tensor)
            grad_numpy = grad.cpu().detach().numpy().squeeze()
            return np.asarray(grad_numpy, dtype=np.float64)  # Ensure float64 for SLSQP compatibility

        optimize_idx = self._model.output_col.index(field_name)
        obj = partial(self._objective, idx=optimize_idx, optimize_fields=self._model.input_col, fixed_fields={})

        best = None
        constraint = NonlinearConstraint(
            cfield_constraint_fn, 
            -5e-2,  # Allow larger negative deviation
            5e-2,   # Allow larger positive deviation
            jac=cfield_constraint_grad,
            keep_feasible=False  # Remove keep_feasible to avoid warning
        )
        
        def generate_initial_guess():
            # Get denormalized value for better initialization
            sqrt_value = np.sqrt(abs(c_value))  # Using abs to handle negative values safely
            
            # random_value = -1
            random_value = np.random.rand() + 1.0
            # random_value = np.random.rand() * 0.3 + 1.0
            phase_init = random_value * sqrt_value
            stride_init = c_value / phase_init
            
            # Initialize guess array
            initial_guess = np.zeros(len(self._model.input_col))
            
            # Find indices for Phase and Stride
            phase_idx = self._model.input_col.index('Phase')
            stride_idx = self._model.input_col.index('Stride')
            
            # Set Phase and Stride to sqrt of target velocity for balanced initial guess
            param = self.model.normalize({'Phase': phase_init, 'Stride': stride_init})
            
            initial_guess[phase_idx] = param['Phase']
            initial_guess[stride_idx] = param['Stride']
            
            # Fill remaining values with random values
            for i, field in enumerate(self._model.input_col):
                if initial_guess[i] == 0:  # Only fill unset values
                    initial_guess[i] = np.random.uniform(self._param[field][0], self._param[field][1])
            
            return initial_guess
        
        bounds = [self._param[field] for field in self._model.input_col]
        # min_bound = -10
        # max_bound = -1
        # bound_idx = 0
        # bounds[bound_idx] = (max(self._param[self._model.input_col[bound_idx]][0], min_bound), min(self._param[self._model.input_col[bound_idx]][1], max_bound))
        # print(f"bounds:  {bounds}")
        for _ in range(self._trial):
            initial_guess = generate_initial_guess()
            # Try different optimization methods
            methods = ['SLSQP', 'trust-constr']
            for method in methods:
                try:
                    # Different options for different methods
                    if method == 'trust-constr':
                        options = {'maxiter': 200}
                        # trust-constr can handle different dtypes
                        guess = initial_guess
                        bounds_array = bounds
                    else:
                        options = {'ftol': 1e-4, 'maxiter': 200}
                        # SLSQP requires numpy.float64
                        guess = np.asarray(initial_guess, dtype=np.float64)
                        bounds_array = [(np.float64(b[0]), np.float64(b[1])) for b in bounds]
                    
                    result = minimize(
                        obj, 
                        guess, 
                        method=method,
                        jac=True, 
                        bounds=bounds_array, 
                        constraints=[constraint],
                        options=options
                    )
                    if result.success and self._check_optimized(best, result.fun):
                        best = result
                        break
                except Exception as e:
                    print(f"Warning: Optimization method {method} failed: {e}")
                    continue

        if best is None:
            os.system(f'aplay {PROJECT_ROOT / "data/res/alarm.wav"}')
            raise RuntimeError("Optimization failed to find any feasible solution")

        best_constraint = cfield_constraint_fn(best.x)
        print(f'Final: best_constraint: {best_constraint}')

        # Reconstruct the full optimized input, including fixed fields
        best_x_full = {}
        for i, field in enumerate(self._model.input_col):
            best_x_full[field] = best.x[i] if best.x.ndim > 0 else best.x
        best_x_full = self.model.denormalize(best_x_full)
        best_return = self.model.denormalize({field_name: best.fun})[field_name]
        return best_x_full, best_return


    def run_w_product(self, field_name: str, c_fields: List[str], c_value: float) -> Tuple[Dict[str, float], float]:
        """
        Runs the optimization for the given field
        :param field_name: the field to optimize
        :param c_fields: the fields to multiply
        :param c_value: the value to multiply
        :return: the optimized parameters and the expected value
        """
        max_product = np.prod([self._param_max[field] for field in c_fields])
        min_product = np.prod([self._param_min[field] for field in c_fields])
        if c_value < min_product:
            print(f"[Warning] The product({c_value:.4f}) of {c_fields} is less than min product {min_product:.4f}.")
            min_constraint = {field: self._param_min[field] for field in c_fields}
            return self.run(field_name, fixed_fields=min_constraint)
        if c_value > max_product:
            print(f"[Warning] The product({c_value:.4f}) of {c_fields} is greater than max product {max_product:.4f}.")
            max_constraint = {field: self._param_max[field] for field in c_fields}
            return self.run(field_name, fixed_fields=max_constraint)

        def product_constraint(x: np.array) -> float:
            values = {}
            for c_field in c_fields:
                idx = self.model.input_col.index(c_field)
                values[c_field] = x[idx]
            values = self.model.denormalize(values)
            return np.prod(list(values.values())).item() - c_value

        optimize_idx = self._model.output_col.index(field_name)
        obj = partial(self._objective, idx=optimize_idx, optimize_fields=self._model.input_col, fixed_fields={})

        best = None
        constraint = NonlinearConstraint(product_constraint, 0, 0)
        for _ in range(self._trial):
            initial_guess = np.array([np.random.uniform(self._param[field][0], self._param[field][1]) for field in self._model.input_col])
            bounds = [self._param[field] for field in self._model.input_col]
            # min_bound = -1.5
            # max_bound = 20
            # bound_idx = 0
            # bounds[bound_idx] = (max(self._param[self._model.input_col[bound_idx]][0], min_bound), min(self._param[self._model.input_col[bound_idx]][1], max_bound))
            # print(f"bounds:  {bounds}")
            result = minimize(obj, initial_guess, jac=True, bounds=bounds, constraints=[constraint])
            if self._check_optimized(best, result.fun):
                best = result

        # Reconstruct the full optimized input, including fixed fields
        best_x_full = {}
        for i, field in enumerate(self._model.input_col):
            best_x_full[field] = best.x[i] if best.x.ndim > 0 else best.x
        best_x_full = self.model.denormalize(best_x_full)
        best_return = self.model.denormalize({field_name: best.fun})[field_name]
        return best_x_full, best_return


    @property
    def model(self) -> Regression:
        return self._model

    @property
    def param_range(self) -> Dict[str, Tuple[float, float]]:
        out = {}
        for field in self._model.input_col:
            out[field] = (self._param_min[field], self._param_max[field])
        return out
