use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Write},
};

use ndarray::{Array, Array1, Array2, ArrayView1, Axis, s};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{prelude::*, types::PyType};
use serde::{Deserialize, Serialize};

#[pymodule]
mod ctrnn {
    #[pymodule_export]
    use crate::CTRNN;
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CTRNNConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub step_size: f64,
}

impl Default for CTRNNConfig {
    fn default() -> Self {
        Self {
            input_size: 1,
            hidden_size: 1,
            output_size: 1,
            step_size: 0.1,
        }
    }
}

impl CTRNNConfig {
    pub fn build(&self) -> CTRNN {
        CTRNN::new(
            self.input_size,
            self.hidden_size,
            self.output_size,
            self.step_size,
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CTRNNGenome {
    taus: Vec<f64>,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    gains: Vec<f64>,
}

#[pyclass(module = "ctrnn")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CTRNN {
    config: CTRNNConfig,
    total_size: usize,
    taus: Array1<f64>,
    weights: Array2<f64>,
    biases: Array1<f64>,
    gains: Array1<f64>,
    states: Array1<f64>,
    outputs: Array1<f64>,
}

impl CTRNN {
    pub fn config(&self) -> CTRNNConfig {
        self.config
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (
            self.config.input_size,
            self.config.hidden_size,
            self.config.output_size,
        )
    }

    pub fn euler_step(&mut self, inputs: ArrayView1<f64>) -> ArrayView1<f64> {
        let mut net_inputs = self.outputs.dot(&self.weights);
        let mut input_slice = net_inputs.slice_mut(s![..inputs.dim()]);
        input_slice += &inputs;
        let derivatives = (1.0 / &self.taus) * (net_inputs - &self.states);
        self.states = &self.states + self.config.step_size * derivatives;
        self.outputs = Self::sigmoid((&self.gains * (&self.states + &self.biases)).view());
        self.output()
    }

    fn sigmoid(ref x: ArrayView1<f64>) -> Array1<f64> {
        1.0 / (1.0 + (-x).exp())
    }

    fn inverse_sigmoid(ref x: ArrayView1<f64>) -> Array1<f64> {
        const EPS: f64 = 1e-7;
        (x / (1.0 - x) + EPS).ln()
    }

    pub fn to_file(&self, filename: &str) -> io::Result<()> {
        let parameters = SerializableCTRNN::from(self);
        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &parameters)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Failed to serialize json"))?;
        writer.flush()?;
        Ok(())
    }

    pub fn from_file(filename: &str) -> io::Result<Self> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let parameters: SerializableCTRNN = serde_json::from_reader(reader).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "Failed to deserialize json")
        })?;
        Ok(Self::from(parameters))
    }

    pub fn load_genome_json(&mut self, genome: &str) -> serde_json::Result<()> {
        let genome: CTRNNGenome = serde_json::from_str(genome)?;
        self.taus = Array::from(genome.taus);
        self.biases = Array::from(genome.biases);
        self.weights = Array::from_shape_vec(
            (self.total_size, self.total_size),
            genome.weights.into_iter().flatten().collect(),
        )
        .unwrap();
        self.gains = Array::from(genome.gains);
        self.outputs = Array::zeros(self.total_size);
        self.states = Array::zeros(self.total_size);
        Ok(())
    }

    pub fn output(&self) -> ArrayView1<f64> {
        let output_ix = self.total_size - self.config.output_size;
        self.outputs.slice(s![output_ix..])
    }

    pub fn set_states(&mut self, value: Array1<f64>) {
        self.states = value;
        self.outputs = Self::sigmoid((&self.gains * (&self.states + &self.biases)).view());
    }

    pub fn set_outputs(&mut self, value: Array1<f64>) {
        self.outputs = value;
        self.states = Self::inverse_sigmoid(self.outputs.view()) / &self.gains - &self.biases;
    }
}

#[pymethods]
impl CTRNN {
    /// Create a new randomly initialized CTRNN
    ///
    /// ## Arguments
    /// * `input_size` - number of input neurons
    /// * `hidden_size` - number of hidden neurons
    /// * `output_size` - number of output neurons
    /// * `step_size` - step size for Euler integration
    #[new]
    #[pyo3(signature = (
        input_size,
        hidden_size,
        output_size,
        step_size=0.1,
    ))]
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, step_size: f64) -> Self {
        let size = input_size + hidden_size + output_size;

        Self {
            config: CTRNNConfig {
                input_size,
                hidden_size,
                output_size,
                step_size,
            },
            total_size: size,
            taus: Array::ones(size),
            weights: Array::ones((size, size)),
            biases: Array::zeros(size),
            gains: Array::ones(size),
            outputs: Array::zeros(size),
            states: Array::zeros(size),
        }
    }

    pub fn burn_in(&mut self, steps: usize) {
        for _ in 0..steps {
            self.euler_step(Array::zeros(self.total_size).view());
        }
    }

    pub fn reset(&mut self) {
        self.set_states(Array::zeros(self.outputs.dim()));
    }

    // --- glue ---

    #[pyo3(name = "euler_step")]
    fn euler_step_py<'py>(
        &mut self,
        py: Python<'py>,
        inputs: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let nd_inputs = inputs.as_array();
        let output = self.euler_step(nd_inputs);
        output.to_pyarray(py)
    }

    #[pyo3(name = "to_file")]
    fn to_file_py(&self, filename: &str) -> PyResult<()> {
        Ok(self.to_file(filename)?)
    }

    #[classmethod]
    #[pyo3(name = "from_file")]
    fn from_file_py(_cls: &Bound<'_, PyType>, filename: &str) -> PyResult<Self> {
        Ok(Self::from_file(filename)?)
    }

    #[pyo3(name = "load_genome_json")]
    fn load_genome_json_py(&mut self, genome: &str) -> PyResult<()> {
        Ok(self.load_genome_json(genome).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "Failed to deserialize json")
        })?)
    }

    #[pyo3(name = "clone")]
    fn clone_py(&self) -> Self {
        self.clone()
    }

    #[pyo3(name = "get_output")]
    fn get_output_py<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let output = self.output().insert_axis(Axis(0));
        output.to_pyarray(py)
    }

    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        let state = bincode::serialize(self)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Failed to serialize bytes"))?;
        Ok(state)
    }

    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        *self = bincode::deserialize(&state).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "Failed to deserialize bytes")
        })?;
        Ok(())
    }

    fn __getnewargs__(&self) -> PyResult<(usize, usize, usize, f64)> {
        Ok((
            self.config.input_size,
            self.config.hidden_size,
            self.config.output_size,
            self.config.step_size,
        ))
    }

    #[getter(taus)]
    fn taus_py<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.taus.to_pyarray(py)
    }

    #[setter(taus)]
    fn set_taus_py<'py>(&mut self, value: PyReadonlyArray1<'py, f64>) {
        self.taus = value.to_owned_array();
    }

    #[getter(weights)]
    fn weights_py<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.weights.to_pyarray(py)
    }

    #[setter(weights)]
    fn set_weights_py<'py>(&mut self, value: PyReadonlyArray2<'py, f64>) {
        self.weights = value.to_owned_array();
    }

    #[getter(biases)]
    fn biases_py<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.biases.to_pyarray(py)
    }

    #[setter(biases)]
    fn set_biases_py<'py>(&mut self, value: PyReadonlyArray1<'py, f64>) {
        self.biases = value.to_owned_array();
    }

    #[getter(gains)]
    fn gains_py<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.gains.to_pyarray(py)
    }

    #[setter(gains)]
    fn set_gains_py<'py>(&mut self, value: PyReadonlyArray1<'py, f64>) {
        self.gains = value.to_owned_array();
    }

    #[getter(states)]
    fn states_py<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.states.to_pyarray(py)
    }

    #[setter(states)]
    fn set_states_py<'py>(&mut self, value: PyReadonlyArray1<'py, f64>) {
        let value = value.to_owned_array();
        self.set_states(value);
    }

    #[getter(outputs)]
    fn outputs_py<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.outputs.to_pyarray(py)
    }

    #[setter(outputs)]
    fn set_outputs_py<'py>(&mut self, value: PyReadonlyArray1<'py, f64>) {
        let value = value.to_owned_array();
        self.set_outputs(value);
    }
}

// Not a very efficient solution, but easiest to quickly implement
// If it becomes a bottleneck implement custom serialization function
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableCTRNN {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    total_size: usize,
    step_size: f64,
    taus: Vec<f64>,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    gains: Vec<f64>,
    states: Vec<f64>,
    outputs: Vec<f64>,
}

impl From<&CTRNN> for SerializableCTRNN {
    fn from(value: &CTRNN) -> Self {
        Self {
            input_size: value.config.input_size,
            hidden_size: value.config.hidden_size,
            output_size: value.config.output_size,
            total_size: value.total_size,
            step_size: value.config.step_size,
            taus: value.taus.to_vec(),
            weights: value.weights.outer_iter().map(|x| x.to_vec()).collect(),
            biases: value.biases.to_vec(),
            gains: value.gains.to_vec(),
            states: value.states.to_vec(),
            outputs: value.outputs.to_vec(),
        }
    }
}

impl From<SerializableCTRNN> for CTRNN {
    fn from(value: SerializableCTRNN) -> Self {
        let config = CTRNNConfig {
            input_size: value.input_size,
            hidden_size: value.hidden_size,
            output_size: value.output_size,
            step_size: value.step_size,
        };

        Self {
            config,
            total_size: value.total_size,
            taus: Array::from(value.taus),
            weights: Array::from_shape_vec(
                (value.total_size, value.total_size),
                value.weights.into_iter().flatten().collect(),
            )
            .unwrap(),
            biases: Array::from(value.biases),
            gains: Array::from(value.gains),
            states: Array::from(value.states),
            outputs: Array::from(value.outputs),
        }
    }
}

// /// Error for bad values
// struct ValueError(String);

// impl From<ValueError> for PyErr {
//     fn from(value: ValueError) -> Self {
//         PyValueError::new_err(value.0)
//     }
// }
