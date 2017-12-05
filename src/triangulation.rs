//
// libskry_r - astronomical image stacking
// Copyright (c) 2017 Filip Szczerek <ga.software@yahoo.com>
//
// This project is licensed under the terms of the MIT license
// (see the LICENSE file for details).
//
//
// File description:
//   Delaunay triangulation.
//

use defs::{Point, Rect};


//TODO: change to usize::max_value(); once const functions are available on stable
const EMPTY: usize = 1 << 30; 


#[derive(Copy, Clone, Default)]
pub struct Edge {
    /// First vertex
    pub v0: usize,
    /// Second vertex
    pub v1: usize,
    
    /// First adjacent triangle (if EMPTY, `t1` is not EMPTY)
    pub t0: usize,
    /// Second adjacent triangle (if EMPTY, `t0` is not EMPTY)
    pub t1: usize, 

    /// First opposite vertex (if EMPTY, `w1` is not EMPTY)
    pub w0: usize,
    /// Second opposite vertex (if EMPTY, `w0` is not EMPTY)
    pub w1: usize 
}


/// Vertices and edges are specified in CCW order.
///
/// Note: in `Triangulation.edges` the edges' vertices may not be specified in the order
/// mentioned below for `e0`, `e1`, `e2`.
///
#[derive(Copy, Clone, Default)]
pub struct Triangle {
    /// First vertex 
    pub v0: usize,
    /// Second vertex
    pub v1: usize,
    /// Third vertex
    pub v2: usize,

    /// First edge (contains v0, v1)
    pub e0: usize,
    /// Second edge (contains v1, v2)
    pub e1: usize,
    /// Third edge (contains v2, v0)
    pub e2: usize
}


impl Triangle {
    pub fn contains(&self, vertex: usize) -> bool {
        vertex == self.v0 ||  
        vertex == self.v1 ||
        vertex == self.v2
    }
    
    
    pub fn next_vertex(&self, vertex: usize) -> usize {
        if vertex == self.v0 {
            return self.v1;
        } else if vertex == self.v1 {
            return self.v2;
        } else if vertex == self.v2 {
            return self.v0;
        } else {
            panic!("Attempted to get next vertex after {} in triangle ({}, {}, {}).",
                vertex, self.v0, self.v1, self.v2);
        }
    }
    
    
    /// Returns the 'leading' edge of a vertex.
    ///
    /// Each vertex has a 'leading' and a 'trailing' edge (corresponding to CCW order).
    /// The 'leading' edge is the one which contains the vertex and a vertex which succeeds it in CCW order.
    ///
    pub fn get_leading_edge_containing_vertex(&self, vertex: usize) -> usize {
        if vertex == self.v0 {
            return self.e0;
        } else if vertex == self.v1 {
            return self.e1;
        } else if vertex == self.v2 {
            return self.e2;
        } else {
            panic!("Attempted to get leading edge containing vertex {} in triangle ({}, {}, {}).",
                vertex, self.v0, self.v1, self.v2);
        }
    }
}


#[derive(Clone, Default)]
pub struct Triangulation {
    vertices:  Vec<Point>,
    edges:     Vec<Edge>,
    triangles: Vec<Triangle>
}


impl Triangulation {
    pub fn get_vertices(&self) -> &[Point] {
        &self.vertices[..]
    }
 
    
    pub fn get_edges(&self) -> &[Edge] {
        &self.edges[..]
    }
 
    
    pub fn get_triangles(&self) -> &[Triangle] {
        &self.triangles[..]
    }
 
    
    /// Checks if vertex 'pidx' lies inside triangle 'tidx'.
    pub fn is_inside_triangle(&self, pidx: usize, tidx: usize) -> bool {
        let t = &self.triangles[tidx];
        let (u, v) = calc_barycentric_coords!(&self.vertices[pidx],
                                              &self.vertices[t.v0],
                                              &self.vertices[t.v1],
                                              &self.vertices[t.v2]);
        
        let w = 1.0 - u - v;
        
        u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0 && w >= 0.0 && w <= 1.0  
    }
    
    /// Finds Delaunay triangulation for the specified point set (all have to be different).
    ///
    /// Also adds three additional points for the initial triangle which covers the whole set
    /// and `envelope`. `Envelope` has to contain all `points`.
    ///
    pub fn find_delaunay_triangulation(points: &[Point], envelope: &Rect) -> Triangulation {
        let mut tri = Triangulation{ vertices: vec![], edges: vec![], triangles: vec![] };

        tri.vertices.extend_from_slice(points);
    
        // Create the initial triangle which covers `envelope` (which in turn must contain all of `points`);
        // append its vertices `all_` at the end of the array
    
        let all0 = Point{ x: envelope.x - 15*(envelope.height as i32)/10 - 16,
                          y: envelope.y - (envelope.height as i32)/10 - 16 };
        
        let all1 = Point{ x: envelope.x + (envelope.width + 15*envelope.height/10 + 16) as i32,
                          y: all0.y };

        let all2 = Point{ x: envelope.x + (envelope.width as i32)/2,
                          y: envelope.y + (envelope.height + 15*envelope.width/10 + 16) as i32 };

        tri.vertices.push(all0);
        tri.vertices.push(all1);
        tri.vertices.push(all2);
    
        // Initial triangle's edges
        
        tri.edges.push(Edge{ v0: points.len() + 0, v1: points.len() + 1,
                             t0: 0, t1: EMPTY,
                             w0: points.len() + 2, w1: EMPTY });
        
        tri.edges.push(Edge{ v0: points.len() + 1, v1: points.len() + 2,
                             t0: 0, t1: EMPTY,
                             w0: points.len() + 0, w1: EMPTY });
    
        tri.edges.push(Edge{ v0: points.len() + 2, v1: points.len() + 0,
                             t0: 0, t1: EMPTY,
                             w0: points.len() + 1, w1: EMPTY });
    
        tri.triangles.push(Triangle{ v0: points.len() + 0,
                                     v1: points.len() + 1,
                                     v2: points.len() + 2,
                                     e0: 0, e1: 1, e2: 2 });

        // Process subsequent points and incrementally refresh the triangulation
        for pidx in 0..points.len() {
            // 1) Find an existing triangle 't' with index 'tidx' to which 'pidx' belongs
            let mut tidx: usize = EMPTY;
            
            for j in 0..tri.triangles.len() {
                if tri.is_inside_triangle(pidx, j) {
                    tidx = j;
                    break;
                }
            }
            assert!(tidx != EMPTY); // Will never happen, unless 'envelope' does not contain all 'points'

            // Check if point 'pidx' belongs to one of triangle 'tidx's edges
            let mut insertion_edge = EMPTY;
            
            let mut edges_to_check: [usize; 3] = [0; 3]; {
                let t = &tri.triangles[tidx];
                
                // All items in 'points' have to be different
                assert!(tri.vertices[t.v0] != points[pidx] &&
                        tri.vertices[t.v1] != points[pidx] &&
                        tri.vertices[t.v2] != points[pidx]);


                edges_to_check.copy_from_slice(&[t.e0, t.e1, t.e2]);
            }

            for i in 0..3 {
                if point_belongs_to_line(&points[pidx], &tri.vertices[tri.edges[edges_to_check[i]].v0],
                                                        &tri.vertices[tri.edges[edges_to_check[i]].v1]) {
                                                            
                    insertion_edge = edges_to_check[i];
                    break;
                }
            }
            
            if insertion_edge != EMPTY {
                tri.add_point_on_edge(pidx, insertion_edge);                
            } else {
                tri.add_point_inside_triangle(pidx, tidx);
            }
        }
 
        tri  
    }
    
    /// Adds new point `pidx` that lies on an existing edge `eidx`.
    fn add_point_on_edge(&mut self, pidx: usize, eidx: usize) {
        //    Starting configuration: (| = edge 'e')
        //
        //                k0
        //               .|.
        //              . | .
        //            q0  |  .
        //            .   |   q3
        //           .    |    .
        //         wt0    p     wt1
        //          .  t0 | t1  .
        //           .    |    .
        //           q1   |   q2
        //             .  |  .
        //              . | .
        //               k1
        //  
        //
        //    Point 'p' is inserted into edge 'e', which has adjacent triangles t0, t1
        //    and the corresponding opposing vertices wt0, wt1. The adjacent triangles
        //    form a quadrilateral (with edges q0-3) whose diagonal is the edge 'e'.
        //
        //    Edge 'e' is subdivided into e0, e1, where e0 contains v0 and e1 contains v1.
        //    This creates two more edges e2, e3, which subdivide triangle t0 into triangles t0a/t0b
        //    and triangle t1 into t1a/t1b:
        //
        //                k0
        //               .|.
        //              . | .
        //            q0  |  .
        //            .   e0  . q3
        //           .    |    .
        //          . t0a | t1b .
        //         .      |      .
        //      wt0...e2..p...e3..wt1
        //        .       |       .
        //         .  t0b |      .
        //          .     | t1a .
        //           .    |    .
        //           q1   e1  q2
        //             .  |  .
        //              . | .
        //               k1
        //
        //    After subdivision, the Delaunay condition needs to be checked for edges e0..3, q0..3.
    

        // We subdivide the 2 triangles adjacent to 'e' into two new triangles each. Old triangles in the array
        // will be reused, so make room for just 2 new ones.
        
        self.triangles.push(Default::default());
        self.triangles.push(Default::default());
        
        // We subdivide 'e' into two edges at point 'pidx' and create 2 more edges connected to 'pidx'.
        // The array element storing 'e' will be reused, so make room for just 3 new ones.
        
        self.edges.push(Default::default());
        self.edges.push(Default::default());
        self.edges.push(Default::default());
        
        macro_rules! e { () => { self.edges[eidx] } }
    
        let t0_idx = self.edges[eidx].t0;
        macro_rules! t0 { () => { self.triangles[t0_idx] } }
        let t1_idx = self.edges[eidx].t1;
        macro_rules! t1 { () => { self.triangles[t1_idx] } }
    
        let wt0_idx; // Vertex opposite of 'e' belonging to 't0'
        let wt1_idx; // Vertex opposite of 'e' belonging to 't1'
        
        if t0!().contains(self.edges[eidx].w0) {
            wt0_idx = e!().w0;
            wt1_idx = e!().w1;
        } else {
            wt0_idx = e!().w1;
            wt1_idx = e!().w0;
        }
    
        let k0 = t1!().next_vertex(wt1_idx);
        let k1 = t0!().next_vertex(wt0_idx);
    
        // Edges of the quadrilateral (k0, wt0_idx, k1, wt1_idx);
        // q0 contains (k0, wt0_idx); q1, q2, q3 are the subsequent edges in CCW order.
        
        let q0_idx = t0!().get_leading_edge_containing_vertex(k0);
        let q1_idx = t0!().get_leading_edge_containing_vertex(wt0_idx);
        let q2_idx = t1!().get_leading_edge_containing_vertex(k1);
        let q3_idx = t1!().get_leading_edge_containing_vertex(wt1_idx);
        
        // 't0' gets subdivided into 't0a', 't0b'
        // 't0a' reuses the storage of 't0'
        let t0a_idx = e!().t0;
        let mut t0a: Triangle = Default::default();
                
        let t0b_idx = self.triangles.len() - 2; // the 1st of the newly allocated triangles
        let t1b_idx = self.triangles.len() - 1; // the 2nd of the newly allocated triangles
        
        // 't1' gets subdivided into 't1a', 't1b'
        // 't1a' reuses the storage of 't1'
        let t1a_idx = e!().t1;
        let mut t1a: Triangle = Default::default();
        
        // Edge 'e' (of 'eidx') gets subdivided into 'e0' and 'e1'. New edge 'e2' belongs to 't0',
        // new edge 'e3' belongs to 't1'.

        let e0_idx = eidx;
        let e1_idx = self.edges.len() - 3;
        let e2_idx = self.edges.len() - 2;
        let e3_idx = self.edges.len() - 1;
                     
        let mut e0: Edge = Default::default();

        e0.v0 = pidx;
        e0.v1 = k0;
        e0.t0 = t0a_idx;
        e0.t1 = t1b_idx;
        e0.w0 = wt0_idx;
        e0.w1 = wt1_idx;
    
        {
            let e1 = &mut self.edges[e1_idx];            
            
            e1.v0 = pidx;
            e1.v1 = k1;
            e1.t0 = t0b_idx;
            e1.t1 = t1a_idx;
            e1.w0 = wt0_idx;
            e1.w1 = wt1_idx;
        }

        {
            let e2 = &mut self.edges[e2_idx];    
            e2.v0 = pidx;
            e2.v1 = wt0_idx;
            e2.t0 = t0a_idx;
            e2.t1 = t0b_idx;
            e2.w0 = k0;
            e2.w1 = k1;
        }
        
        {
            let e3 = &mut self.edges[e3_idx];
                
            e3.v0 = pidx;
            e3.v1 = wt1_idx;
            e3.t0 = t1a_idx;
            e3.t1 = t1b_idx;
            e3.w0 = k0;
            e3.w1 = k1;
        }
    
        t0a.v0 = pidx;
        t0a.v1 = k0;
        t0a.v2 = wt0_idx;
        t0a.e0 = e0_idx;
        t0a.e1 = q0_idx;
        t0a.e2 = e2_idx;
    
        {
            let t0b = &mut self.triangles[t0b_idx];
            
            t0b.v0 = pidx;
            t0b.v1 = wt0_idx;
            t0b.v2 = k1;
            t0b.e0 = e2_idx;
            t0b.e1 = q1_idx;
            t0b.e2 = e1_idx;
        }
    
        t1a.v0 = pidx;
        t1a.v1 = k1;
        t1a.v2 = wt1_idx;
        t1a.e0 = e1_idx;
        t1a.e1 = q2_idx;
        t1a.e2 = e3_idx;
    
        {
            let t1b = &mut self.triangles[t1b_idx];
            
            t1b.v0 = pidx;
            t1b.v1 = wt1_idx;
            t1b.v2 = k0;
            t1b.e0 = e3_idx;
            t1b.e1 = q3_idx;
            t1b.e2 = e0_idx;
        }
      
        
        // Update the edges of the quadrilateral (e.v0, wt0_idx, e.v2, wt1_idx): their adjacent triangles and opposite vertices

        replace_adjacent_triangle(&mut self.edges[q0_idx], t0_idx, t0a_idx);
        replace_opposing_vertex(&mut self.edges[q0_idx], k1, pidx);
    
        replace_adjacent_triangle(&mut self.edges[q1_idx], t0_idx, t0b_idx);
        replace_opposing_vertex(&mut self.edges[q1_idx], k0, pidx);
    
        replace_adjacent_triangle(&mut self.edges[q2_idx], t1_idx, t1a_idx);
        replace_opposing_vertex(&mut self.edges[q2_idx], k0, pidx);
    
        replace_adjacent_triangle(&mut self.edges[q3_idx], t1_idx, t1b_idx);
        replace_opposing_vertex(&mut self.edges[q3_idx], k1, pidx);
    
        // Overwrite old triangles and edges
        self.triangles[t0_idx] = t0a;
        self.triangles[t1_idx] = t1a;
        e!() = e0;
    
        // Check Delaunay condition for all affected edges
        for i in [e0_idx, e1_idx, e2_idx, e3_idx,
                  q0_idx, q1_idx, q2_idx, q3_idx].iter() {
                      
            self.test_and_swap_edge(*i, EMPTY, EMPTY);
        }
    }

    
    /// Adds a new point 'pidx' inside an existing triangle 'tidx'.
    fn add_point_inside_triangle(&mut self, pidx: usize, tidx: usize) {
        //        
        // Subdivide 't' into 3 sub-triangles 'tsub0', 'tsub1', 'tsub3' using 'pidx'
        //
        // The order of existing triangles has to be preserved (they are referenced by the existing edges),
        // so replace 't' by 'tsub0' and add 'tsub1' and 'tsub2' at the triangle array's end.
    
        let mut tsub0: Triangle = Default::default();
        let tsub0idx = tidx;
    
        // Add 2 new triangles
        self.triangles.push(Default::default());
        self.triangles.push(Default::default());
        
        let tsub1idx = self.triangles.len() - 2;
        let tsub2idx = self.triangles.len() - 1;
    
        macro_rules! tsub1 { () => { self.triangles[tsub1idx] } }
        macro_rules! tsub2 { () => { self.triangles[tsub2idx] } }
    
        macro_rules! t { () => { self.triangles[tidx] } }
    
        // Add 3 new edges 'enew0', 'enew1', 'enew2' which connect 't.v0', 't.v1', 't.v2' with 'pidx'

        self.edges.push(Edge{ v0: t!().v0,
                              v1: pidx,
                              t0: tsub0idx,
                              t1: tsub2idx,
                              w0: t!().v1,
                              w1: t!().v2 });
        let enew0 = self.edges.len() - 1;
        
        self.edges.push(Edge{ v0: t!().v1,
                              v1: pidx,
                              t0: tsub0idx,
                              t1: tsub1idx,
                              w0: t!().v0,
                              w1: t!().v2 });
        let enew1 = self.edges.len() - 1;
    
        //DA_APPEND(tri->edges, ((struct SKRY_edge) { .v0 = t->v2, .v1 = pidx, .t0 = tsub1idx, .t1 = tsub2idx, .w0 = t->v1, .w1 = t->v0 }));
        self.edges.push(Edge{ v0: t!().v2,
                              v1: pidx,
                              t0: tsub1idx,
                              t1: tsub2idx,
                              w0: t!().v1,
                              w1: t!().v0 });
        let enew2 = self.edges.len() - 1;
    
        // Fill the new triangles' data
    
        tsub0.v0 = pidx;
        tsub0.v1 = t!().v0;
        tsub0.v2 = t!().v1;
        tsub0.e0 = enew0;
        tsub0.e1 = t!().e0;
        tsub0.e2 = enew1;

        tsub1!().v0 = pidx;
        tsub1!().v1 = t!().v1;
        tsub1!().v2 = t!().v2;
        tsub1!().e0 = enew1;
        tsub1!().e1 = t!().e1;
        tsub1!().e2 = enew2;
    
        tsub2!().v0 = pidx;
        tsub2!().v1 = t!().v2;
        tsub2!().v2 = t!().v0;
        tsub2!().e0 = enew2;
        tsub2!().e1 = t!().e2;
        tsub2!().e2 = enew0;
    
        // Update adjacent triangle and opposing vertex data for 't's edges
    
        replace_opposing_vertex(&mut self.edges[t!().e0], t!().v2, pidx);
        replace_adjacent_triangle(&mut self.edges[t!().e0], tidx, tsub0idx);
    
        replace_opposing_vertex(&mut self.edges[t!().e1], t!().v0, pidx);
        replace_adjacent_triangle(&mut self.edges[t!().e1], tidx, tsub1idx);
    
        replace_opposing_vertex(&mut self.edges[t!().e2], t!().v1, pidx);
        replace_adjacent_triangle(&mut self.edges[t!().e2], tidx, tsub2idx);
    
        // Keep note of the 't's edges for the subsequent Delaunay check
        let te0 = t!().e0;
        let te1 = t!().e1;
        let te2 = t!().e2;
    
        // Original triangle 't' is no longer needed, replace it with 'tsub0'
        t!() = tsub0;
    
        // 3) Check Delaunay condition for the old 't's edges and swap them if necessary.
        //    Also recursively check any edges affected by the swap.
    
        self.test_and_swap_edge(te0, enew0, enew1);
        self.test_and_swap_edge(te1, enew1, enew2);
        self.test_and_swap_edge(te2, enew2, enew0);
    }
    
    
    /// Checks if point `pidx` is inside the triangle's `tidx` circumcircle.
    #[allow(non_snake_case)]
    fn is_inside_circumcircle(&self, pidx: usize, tidx: usize) -> bool {
        let p = &self.vertices[pidx];
        let t = &self.triangles[tidx];
        
        let A = &self.vertices[t.v0];
        let B = &self.vertices[t.v1];
        let C = &self.vertices[t.v2];
    
    
        // Coordinates of the circumcenter
        let ux: f32;
        let uy: f32;
        
        // Squared radius of the circumcircle
        let radiusq: f32;
    
        // Note: the formulas below work correctly regardless of the handedness of the coordinate system used
        //       (for triangulation of points in an image a left-handed system is used here, i.e. X grows to the right, Y downwards)
    
        let d = 2.0 * (A.x as f32 * (B.y - C.y) as f32 +
                       B.x as f32 * (C.y - A.y) as f32 +
                       C.x as f32 * (A.y - B.y) as f32);
    
        if d.abs() > 1.0e-8 {
            ux = ((sqr!(A.x as f32) + sqr!(A.y as f32)) * (B.y - C.y) as f32 +
                  (sqr!(B.x as f32) + sqr!(B.y as f32)) * (C.y - A.y) as f32 +
                  (sqr!(C.x as f32) + sqr!(C.y as f32)) * (A.y - B.y) as f32) / d;

            uy = ((sqr!(A.x as f32) + sqr!(A.y as f32)) * (C.x - B.x) as f32 +
                  (sqr!(B.x as f32) + sqr!(B.y as f32)) * (A.x - C.x) as f32 +
                  (sqr!(C.x as f32) + sqr!(C.y as f32)) * (B.x - A.x) as f32) / d;
    
            radiusq = sqr!(ux - A.x as f32) + sqr!(uy - A.y as f32);
        } else {
             // Degenerated triangle (co-linear vertices)
            let dist_AB_sq = sqr!((A.x - B.x) as f32) + sqr!((A.y - B.y) as f32);
            let dist_AC_sq = sqr!((A.x - C.x) as f32) + sqr!((A.y - C.y) as f32);
            let dist_BC_sq = sqr!((B.x - C.x) as f32) + sqr!((B.y - C.y) as f32);
    
            // Extreme vertices of the degenerated triangle
            //struct SKRY_point *ext1, *ext2;
            let ext1: &Point;
            let ext2: &Point;
            
            if dist_AB_sq >= dist_AC_sq && dist_AB_sq >= dist_BC_sq {
                ext1 = &A;
                ext2 = &B;
            } else if dist_AC_sq >= dist_AB_sq && dist_AC_sq >= dist_BC_sq {
                ext1 = &A;
                ext2 = &C;
            } else {
                ext1 = &B;
                ext2 = &C;
            }
    
            ux = (ext1.x + ext2.x) as f32 * 0.5;
            uy = (ext1.y + ext2.y) as f32 * 0.5;
    
            radiusq = 0.25 * (sqr!((ext1.x - ext2.x) as f32) + sqr!((ext1.y - ext2.y) as f32));
        }
    
        sqr!(p.x as f32 - ux) + sqr!(p.y as f32 - uy) < radiusq
    }
    
    
    
    /// Ensures the specified edge satisfies the Delaunay condition.
    ///
    /// If edge `e` violates the Delaunay condition, swaps it and recursively
    /// continues to test the 4 neighboring edges.
    ///
    /// Before:
    ///
    ///     v3--e2---v2
    ///    / t1 ___/ /
    ///  e3  __e4   e1
    ///  / _/   t0 /
    /// v0---e0--v1
    ///
    /// After swapping e4:
    ///
    ///     v3--e2---v2
    ///    / \   t0 /
    ///  e3   e4   e1
    ///  /  t1 \  /
    /// v0--e0--v1
    ///
    /// How to decide which of the new triangles is now `t0` and which `t1`?
    /// For each of the triangles adjacent to `e4` before the swap, take their vertex opposite to `e4` and the next vertex.
    /// After edge swap, the new triangle still contains the same 2 vertices. From the example above:
    ///
    /// 1) For `t0` (`v0-v1-v2`), use `v1`, `v2`. After swap, the new `t0` is the triangle which still contains `v1`, `v2`.
    /// 2) For `t1` (`v0-v2-v3`), use `v3`, `v0`. After swap, the new `t1` is the triangle which still contains `v3`, `v0`.
    ///
    /// After the swap is complete, recursively test `e0`, `e1`, `e2` and `e3`.
    ///
    fn test_and_swap_edge(&mut self,
                          e: usize,
                          // Edges to skip when checking what needs swapping (may be SKRY_EMPTY)
                          eskip1: usize,
                          eskip2: usize) {
                              
        // Edge 'e' before the swap
        let eprev: Edge = self.edges[e];
    
        // 0) Check the Delaunay condition for 'e's adjacent triangles
    
        if self.edges[e].t0 == EMPTY || self.edges[e].t1 == EMPTY {
            return;
        }

        // Triangles which share edge 'e' before the swap
        let t0prev = eprev.t0;
        let t1prev = eprev.t1;
    
        //TODO: guarantee that always 'w0' belongs to 't0' and 'w1' to 't1' - then we can get rid of all the "contains" checks below
    
        let mut swap_needed = false;
    
        if !swap_needed && self.triangles[t0prev].contains(eprev.w0) && self.is_inside_circumcircle(eprev.w1, eprev.t0) {
            swap_needed = true;
        }
    
        if !swap_needed && self.triangles[t0prev].contains(eprev.w1) && self.is_inside_circumcircle(eprev.w0, eprev.t0) {
            swap_needed = true;
        }
    
        if !swap_needed && self.triangles[t1prev].contains(eprev.w0) && self.is_inside_circumcircle(eprev.w1, eprev.t1) {
            swap_needed = true;
        }
    
        if !swap_needed && self.triangles[t1prev].contains(eprev.w1) && self.is_inside_circumcircle(eprev.w0, eprev.t1) {
            swap_needed = true;
        }
    
        if !swap_needed {
            return;
        }
    
        // List of at most 4 edges that have to be checked recursively after 'e' is swapped
        // FIXME: do we have to check all the 4 neighboring edges?

        let mut num_edges_to_check: usize = 0;
        let mut edges_to_check: [usize; 4] = [0; 4]; 
    
        if self.triangles[t0prev].e0 != e && self.triangles[t0prev].e0 != eskip1 && self.triangles[t0prev].e0 != eskip2 {
            edges_to_check[num_edges_to_check] = self.triangles[t0prev].e0;
            num_edges_to_check += 1;
        }
        
        if self.triangles[t0prev].e1 != e && self.triangles[t0prev].e1 != eskip1 && self.triangles[t0prev].e1 != eskip2 {
            edges_to_check[num_edges_to_check] = self.triangles[t0prev].e1;
            num_edges_to_check += 1;
        }
        
        if self.triangles[t0prev].e2 != e && self.triangles[t0prev].e2 != eskip1 && self.triangles[t0prev].e2 != eskip2 {
            edges_to_check[num_edges_to_check] = self.triangles[t0prev].e2;
            num_edges_to_check += 1;
        }
    
        if self.triangles[t1prev].e0 != e && self.triangles[t1prev].e0 != eskip1 && self.triangles[t1prev].e0 != eskip2 {
            edges_to_check[num_edges_to_check] = self.triangles[t1prev].e0;
            num_edges_to_check += 1;
        }
        
        if self.triangles[t1prev].e1 != e && self.triangles[t1prev].e1 != eskip1 && self.triangles[t1prev].e1 != eskip2 {
            edges_to_check[num_edges_to_check] = self.triangles[t1prev].e1;
            num_edges_to_check += 1;
        }
        
        if self.triangles[t1prev].e2 != e && self.triangles[t1prev].e2 != eskip1 && self.triangles[t1prev].e2 != eskip2 {
            edges_to_check[num_edges_to_check] = self.triangles[t1prev].e2;
            num_edges_to_check += 1;
        }
    
        // 1) Determine the reference vertices for each triangle
        //  
        //      
        //  
        //          D---------C
        //         / t1 ___/ /
        //        /  __e    /
        //       / _/   t0 /
        //      A---------B
        //  
        //      B becomes t0refV
        //      D becomes t1refV
        //
        
    
        let t0refv; // The only vertex in 't0' which does not belong to 'e'
        let t1refv; // The only vertex in 't1' which does not belong to 'e'
    
        if self.triangles[t0prev].contains(eprev.w0) {
            t0refv = eprev.w0;
            t1refv = eprev.w1;
        } else {
            t0refv = eprev.w1;
            t1refv = eprev.w0;
        }
    
        // 2) Update the triangles
    
        let mut t0new: Triangle = Default::default();
        let mut t1new: Triangle = Default::default();
    
        // For each of the new triangles, the reference vertex from step 1) and the next vertex stay the same as before the swap.
        // The third vertex (i.e. the one "previous" to the reference vertex) becomes the vertex opposite 'e', i.e. the other triangle's reference vertex.
        // Additionally, reorder the vertices such that the reference vertex becomes v0.
        // 
        //    
        // 
        //    Before:                    After:
        // 
        //        D---------C           D-------C
        //       / t1 ___/ /           / \  t0 /
        //      /  __e    /           /   e   /
        //     / _/   t0 /           / t1  \ /
        //    A---------B           A-------B
        // 
        //    t0: v0=A, v1=B, v2=C    t0: v0=B, v1=C, v2=D
        //    t1: v0=D, v1=A, v2=C    t1: v0=D, v1=A, v2=B
    
        t0new.v0 = t0refv;
        t0new.v1 = self.triangles[t0prev].next_vertex(t0refv);
        t0new.v2 = t1refv;
    
        t1new.v0 = t1refv;
        t1new.v1 = self.triangles[t1prev].next_vertex(t1refv);
        t1new.v2 = t0refv;
    
        // For each new triangle, update their edges. The 'leading' edge of the reference vertex (now: e0) stays the same. The second edge comes
        // from the other triangle. The third edge is the new 'e' (after the swap).
    
        t0new.e0 = self.triangles[t0prev].get_leading_edge_containing_vertex(t0new.v0);
        t0new.e1 = self.triangles[t1prev].get_leading_edge_containing_vertex(t0new.v1);
        t0new.e2 = e;
    
        t1new.e0 = self.triangles[t1prev].get_leading_edge_containing_vertex(t1new.v0);
        t1new.e1 = self.triangles[t0prev].get_leading_edge_containing_vertex(t1new.v1);
        t1new.e2 = e;
    
        // For each of the 5 edges involved, update their adjacent triangles and opposing vertices information.
        // For 'e' after swap update also the end vertices.
    
        replace_opposing_vertex(&mut self.edges[t0new.e0], t1new.v1, t0new.v2);
        replace_opposing_vertex(&mut self.edges[t0new.e1], t1new.v1, t0new.v0);
        replace_opposing_vertex(&mut self.edges[t1new.e0], t0new.v1, t1new.v2);
        replace_opposing_vertex(&mut self.edges[t1new.e1], t0new.v1, t1new.v0);
    
        replace_adjacent_triangle(&mut self.edges[t0new.e1], eprev.t1, eprev.t0);
        replace_adjacent_triangle(&mut self.edges[t1new.e1], eprev.t0, eprev.t1);
    
        // update edge 'e' after swap
        self.edges[e].w0 = t0new.v1;
        self.edges[e].w1 = t1new.v1;
    
        self.edges[e].v0 = t0new.v0;
        self.edges[e].v1 = t1new.v0;
    
        // Overwrite the old triangles
        
        self.triangles[t0prev] = t0new;
    
        self.triangles[t1prev] = t1new;
    
        // Recursively check the affected edges
        for i in 0..num_edges_to_check {
            self.test_and_swap_edge(edges_to_check[i], e, EMPTY);
        }
    }
    
}


/// Checks if point `p` belong to the line specified by `v0`, `v1`.
fn point_belongs_to_line(p: &Point, v0: &Point, v1: &Point) -> bool {
    0 == -v1.x*v0.y + p.x*v0.y +
         v0.x*v1.y - p.x*v1.y -
         v0.x*p.y + v1.x*p.y
}


fn replace_opposing_vertex(edge: &mut Edge, wold: usize, wnew: usize) {
    if edge.w0 == wold {
        edge.w0 = wnew;
    } else if edge.w1 == wold {
        edge.w1 = wnew;
    } else if edge.w0 == EMPTY {
        edge.w0 = wnew;
    } else if edge.w1 == EMPTY {
        edge.w1 = wnew;
    }
}


fn replace_adjacent_triangle(edge: &mut Edge, told: usize, tnew: usize) {
    if edge.t0 == told {
        edge.t0 = tnew;
    } else if edge.t1 == told {
        edge.t1 = tnew;
    } else if edge.t0 == EMPTY {
        edge.t0 = tnew;
    } else if edge.t1 == EMPTY {
        edge.t1 = tnew;
    }
}
