use rand::Rng; // this is an external dep
use std::env;
use std::error::Error;
use std::time::Instant;
use rayon::prelude::*;

const MAX_LOOPS: usize = 100;

fn compute_centroids(
    xs: &Vec<Vec<f64>>,
    centroid_indicator: &Vec<usize>,
    k : usize,
) -> Vec<Vec<f64>> {
    let n_dim = xs[0].len();
    let mut sums = vec![vec![0.0;n_dim]; k];
    let mut counts = vec![0.0;k];
    xs.iter().zip(centroid_indicator.iter()).for_each(|(x, centroid_x)|{
       x.iter().enumerate().for_each(|(i, val)|{
           sums[*centroid_x][i] += *val;
       });
       counts[*centroid_x] += 1.0;
       // let mut centroid_sum = &mut sums[centroid_x];
       // x.iter().zip(centroid_sum.iter_mut()).for_each(|(val, sum)|{
       //     *sum += *val;
       // });

    });

    sums.iter_mut().zip(counts.iter()).for_each(|(sum, count)|{
        for s in sum {
            *s /= count;
        }
    });

    sums
}

fn dist(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
    unimplemented!()
}

fn l2_norm(p1: &Vec<f64>, p2: &Vec<f64>) -> f64 {
    p1.iter().zip(p2.iter()).map(|(x,y)|{
        let diff = x - y;
        diff*diff       
    }).sum::<f64>().sqrt()
}

/// Determine to which centroid each datum belongs and return whether any
/// indicator has changed.
fn compute_indicators(
    xs: &Vec<Vec<f64>>,
    centroids: &Vec<Vec<f64>>,
    mut centroid_indicator: Vec<usize>
) -> (Vec<usize>, bool) {
    use std::sync::atomic::{AtomicBool, Ordering};

    let mut has_changed = AtomicBool::new(false);

    xs.par_iter()
        .zip(centroid_indicator.par_iter_mut())
        .for_each(|(val, mut ind)| {
            let (_, index) = centroids
                .iter()
                .map(|x| l2_norm(val, x))
                .enumerate()
                .fold((std::f64::INFINITY, 0), |(min, argmin), (ix, dist)| { //(accumulator, iterator)
                    if dist < min {
                        (dist, ix)
                    } else {
                        (min, argmin)
                    }
                });

            if *ind != index {
                has_changed.store(true, Ordering::Relaxed);
            }

            *ind = index;
        });

    (centroid_indicator, has_changed.load(Ordering::Relaxed)) 
}

fn kmeans2(data: &Vec<Vec<f64>>, k: usize) -> Vec<usize> {
    let n = data.len();
    let d = data[0].len();

    let mut centroid_indicator = (0..k).cycle().take(n).collect::<Vec<usize>>();

    let mut means_time: f64 = 0.0;
    let mut ind_time: f64 = 0.0;

    let mut loop_count: usize = 0;
    let result = std::iter::repeat(0)
        .take(MAX_LOOPS)
        .fold((centroid_indicator, false), |(centroid_indicator, done), _| {
            if done {
                (centroid_indicator, done)
            } else {
                let t_start = Instant::now();
                let centroids = compute_centroids(data, &centroid_indicator, k);
                means_time += t_start.elapsed().as_secs_f64();

                let t_start = Instant::now();
                let (centroid_indicator, indicator_has_changed) = compute_indicators(
                    data,
                    &centroids,
                    centroid_indicator
                );
                ind_time += t_start.elapsed().as_secs_f64();

                (centroid_indicator, !indicator_has_changed)
            }
        }).0;

    println!("t_means: {:.4}, t_ind: {:.4}", means_time, ind_time);

    result
}

fn gen_cluster(offset_x: f64, offset_y: f64, n: usize, rng: &mut impl Rng) -> Vec<Vec<f64>> {
    (0..n)
        .map(|_| {
            let x = rng.gen::<f64>() + offset_x;
            let y = rng.gen::<f64>() + offset_y;
            vec![x, y]
        })
        .collect()
}

fn gen_clustered_data(
    offsets: &Vec<(f64, f64)>,
    n_per_cluster: usize,
    rng: &mut impl Rng
) -> Vec<Vec<f64>> {
    let data: Vec<Vec<f64>> = offsets
        .iter()
        .map(|(offset_x, offset_y)| {
            gen_cluster(*offset_x, *offset_y, n_per_cluster, rng)
        })
        .flatten()
        .collect();

    assert_eq!(data.len(), offsets.len() * n_per_cluster);
    data
}

fn print2d(xs: &Vec<Vec<f64>>) {
    println!("x,y");
    xs.iter().for_each(|x| {
        x.iter().enumerate().for_each(|(i, xi)| {
            print!("{xi}");
            if i == 0 {
               print!(",");
            }
        });
        println!();
    });
}

fn write2d(xs: &Vec<Vec<f64>>, file : &str) -> Result<(), Box<dyn Error>> {
    let mut wtr = csv::Writer::from_path(file).unwrap();
    wtr.write_record(&["x", "y"]);
    xs.iter().for_each(|x| {
        x.iter().enumerate().for_each(|(i, xi)| {
            wtr.write_record(&[{xi.to_string()}]);
        });
    });
    wtr.flush();
    Ok(())
}

// #TODO
// - fill in `compute_indicators`
// - make sure it works and stops early when it needs to (no divide by zero crashing like before)
// - create a command line utility (clap crate, optional; look at `derive`)
//   - allow csv importing (csv crate) assume header
// - together, we will parallelize and benchmark with rayon
//
// # Example
//
// ```
// $ kmeans cluster <FILENAME> <K>
// 0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2
//
// $ kmeans gen <N> <FILENAME>
// ```
fn main() {
    let args: Vec<String> = env::args().collect();

    match args[1].as_str() {
        "gen" => {
            let mut rng = rand::thread_rng();
            let n: usize = args[2].parse().unwrap();
            let file_path = &args[3];
            let xs = gen_clustered_data(
                &vec![(-6.0, 6.0), (0.0, 0.0), (6.0, -6.0)],
                n,
                &mut rng
            );
            write2d(&xs, file_path);
        }
        "cluster" => {
            let clus_str = &args[3];
            let cluster_number: usize = clus_str.parse().unwrap();
            let file_path = &args[2];
            let mut reader = csv::Reader::from_path(file_path).unwrap();

            //"rustify" it later
            let mut data = Vec::new();
            for result in reader.records(){
                let mut record = result.unwrap();
                let mut el = Vec::new();
                for field in &record {
                    el.push(field.parse::<f64>().unwrap());            
                }
                data.push(el);
            }
            
            let t_start = Instant::now();
            let clustering = kmeans2(&data, cluster_number);
            let t_total = t_start.elapsed();
            println!("Clustered in {:.4}s", t_total.as_secs_f64());
            // print!("{:?}", clustering); 
            assert_eq!(clustering.len(), data.len());
        }
        _ => eprintln!("Unknown command.")
    }

}

